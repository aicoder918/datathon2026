"""MLM pretrain a small transformer on all 21k seen bar-sequences (50 bars each),
then extract frozen mean-pooled embeddings per session and paired-diff test
whether adding them to the production CatBoost features helps.

Rationale: prior bar_mlm fine-tune had the head collapse to aggregates. Use
frozen embeddings + tree model instead — different mixing, different probe.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from catboost import CatBoostRegressor

from features import (
    SEED, DATA_DIR, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    fit_template_impacts_sector_multi, build_event_features_sector_multi,
    build_event_features_sector_oof,
    sharpe, shape_positions,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

BARS = 50
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 15
BATCH = 128
LR = 3e-4
MASK_P = 0.15

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_sequences():
    """Return (session_ids, arr[N,50,4]) for all seen sessions (train+pub+priv)."""
    parts = []
    for f, prefix in [
        ("bars_seen_train.parquet", "tr"),
        ("bars_seen_public_test.parquet", "pub"),
        ("bars_seen_private_test.parquet", "priv"),
    ]:
        df = pd.read_parquet(DATA_DIR / f)
        df = df.sort_values(["session", "bar_ix"])
        parts.append((prefix, df))
    # train sessions: 0..999, test sessions are also integer-id'd — build a
    # namespaced id so we can split back out later
    session_ids = []
    all_feats = []
    for prefix, df in parts:
        for sid, g in df.groupby("session", sort=True):
            o = g["open"].to_numpy(dtype=np.float32)
            h = g["high"].to_numpy(dtype=np.float32)
            l = g["low"].to_numpy(dtype=np.float32)
            c = g["close"].to_numpy(dtype=np.float32)
            prev = np.concatenate([[c[0]], c[:-1]])  # use first close as prev-of-0
            feats = np.stack([
                (o - prev) / prev,
                (h - prev) / prev,
                (l - prev) / prev,
                (c - prev) / prev,
            ], axis=-1)  # (50, 4)
            session_ids.append(f"{prefix}:{int(sid)}")
            all_feats.append(feats)
    arr = np.stack(all_feats, axis=0)  # (N, 50, 4)
    return session_ids, arr


print("Loading bar sequences ...")
sids, seqs = load_sequences()
print(f"  {len(sids)} sequences, shape {seqs.shape}")

# Standardize per-feature (global)
mu = seqs.mean(axis=(0, 1), keepdims=True)
sd = seqs.std(axis=(0, 1), keepdims=True) + 1e-8
seqs_std = ((seqs - mu) / sd).astype(np.float32)


class MLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(4, D_MODEL)
        self.pos = nn.Embedding(BARS, D_MODEL)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=4 * D_MODEL,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, 4)  # predict all 4 feats at masked positions
        self.mask_token = nn.Parameter(torch.zeros(D_MODEL))

    def encode(self, x):  # x: (B, 50, 4)
        h = self.in_proj(x)
        pos = self.pos(torch.arange(BARS, device=x.device))
        h = h + pos
        h = self.enc(h)
        return h  # (B, 50, D)

    def forward(self, x, mask):
        # x: (B, 50, 4); mask: (B, 50) bool, True = masked
        h_in = self.in_proj(x)
        pos = self.pos(torch.arange(BARS, device=x.device))
        h_in = h_in + pos
        h_in = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h_in)
        h = self.enc(h_in)
        return self.head(h)  # (B, 50, 4)


model = MLM().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
X_t = torch.from_numpy(seqs_std).to(DEVICE)
N = X_t.shape[0]

print(f"\nPretraining MLM ({sum(p.numel() for p in model.parameters()):,} params) ...")
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(N, device=DEVICE)
    losses = []
    for i in range(0, N, BATCH):
        idx = perm[i:i + BATCH]
        xb = X_t[idx]  # (B, 50, 4)
        B = xb.shape[0]
        m = (torch.rand(B, BARS, device=DEVICE) < MASK_P)
        if m.sum() == 0:
            continue
        pred = model(xb, m)  # (B, 50, 4)
        loss = ((pred[m] - xb[m]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  ep {ep+1:2d}/{EPOCHS}  mlm_mse={np.mean(losses):.5f}")

# --- extract frozen mean-pooled embeddings ---
print("\nExtracting frozen embeddings ...")
model.eval()
embs = []
with torch.no_grad():
    for i in range(0, N, BATCH):
        xb = X_t[i:i + BATCH]
        h = model.encode(xb)  # (B, 50, D)
        embs.append(h.mean(dim=1).cpu().numpy())
embs = np.concatenate(embs, axis=0)  # (N, D_MODEL)
print(f"  emb shape {embs.shape}")

# Map back to session id
emb_df = pd.DataFrame(embs, columns=[f"bemb_{k}" for k in range(D_MODEL)])
emb_df["sid"] = sids
tr_mask = emb_df["sid"].str.startswith("tr:")
emb_tr = emb_df[tr_mask].copy()
emb_tr["session"] = emb_tr["sid"].str.split(":").str[1].astype(int)
emb_tr = emb_tr.set_index("session").drop(columns=["sid"])
print(f"  train emb rows: {len(emb_tr)}")


# --- paired-diff: baseline vs baseline+emb ---
N_SPLITS = 60
HOLDOUT_FRAC = 0.2
ITERS = 52
THRESHOLD_Q = 0.35
HALF = N_SPLITS // 2
CB_KW = dict(iterations=ITERS, learning_rate=0.03, depth=5,
             loss_function="MAE", verbose=False)


def cb_pred(Xd, yd, Xh, seed):
    m = CatBoostRegressor(**CB_KW, random_seed=seed)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


def score_from_pred(pred, vol_h, y_h):
    pos = shape_positions(pred, vol_h, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    return sharpe(pos, y_h)


X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

N_SEEDS = 3
print(f"\nPaired-diff: {N_SPLITS} splits × {N_SEEDS} seeds, baseline vs +emb ...")
base_scores = []
emb_scores = []
for r, (hold_s, dev_s) in enumerate(splits):
    dev_h = headlines_train[headlines_train["session"].isin(dev_s)]
    hold_h = headlines_train[headlines_train["session"].isin(hold_s)]
    dev_b = bars_train[bars_train["session"].isin(dev_s)]
    dev_event = build_event_features_oof(dev_h, dev_b, dev_s)
    split_impacts = fit_template_impacts_multi(dev_h, dev_b)
    hold_event = build_event_features_multi(hold_h, hold_s, split_impacts)
    dev_sec = build_event_features_sector_oof(dev_h, dev_b, dev_s)
    sec_impacts = fit_template_impacts_sector_multi(dev_h, dev_b)
    hold_sec = build_event_features_sector_multi(hold_h, hold_s, sec_impacts)
    Xd = X_base.loc[dev_s].join(dev_event).join(dev_sec)
    Xh = X_base.loc[hold_s].join(hold_event).join(hold_sec)
    Xd_emb = Xd.join(emb_tr.loc[dev_s])
    Xh_emb = Xh.join(emb_tr.loc[hold_s])
    yd = y_full.loc[dev_s].to_numpy(dtype=float)
    yh = y_full.loc[hold_s].to_numpy(dtype=float)
    vh = np.asarray(Xh["vol"].values, dtype=float)

    p_base = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
    p_emb = np.mean([cb_pred(Xd_emb, yd, Xh_emb, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
    base_scores.append(score_from_pred(p_base, vh, yh))
    emb_scores.append(score_from_pred(p_emb, vh, yh))
    if (r + 1) % 10 == 0:
        print(f"  scored {r+1}/{N_SPLITS}")

base = np.asarray(base_scores)
emb = np.asarray(emb_scores)
d = emb - base
se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean() / se if se > 0 else 0.0
dA, dB = d[:HALF], d[HALF:]
seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean() / seA if seA > 0 else 0.0
seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean() / seB if seB > 0 else 0.0
mark = "*" if dA.mean() > 0 and dB.mean() > 0 else " "

print(f"\nbaseline raw    mean={base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")
print(f"+emb (seed{N_SEEDS}) mean={emb.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
      f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")

# Save encoder + embeddings for reuse
out_dir = Path(__file__).resolve().parent.parent / "artifacts"
out_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), out_dir / "bar_mlm_encoder.pt")
emb_df.to_parquet(out_dir / "bar_mlm_embeddings.parquet")
print(f"\nSaved encoder and embeddings to {out_dir}")
