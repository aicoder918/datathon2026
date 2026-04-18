"""Next-momentum predictor: self-supervised MLM on 21k bar sequences, then
supervised fine-tune of a head to predict forward 5-bar return on the 1k
labeled training sessions. Produces ONE scalar per session ("predicted
bar-50..54 return") which we paired-diff as a new CatBoost feature.

Pieces applied from the Kaggle playbook the user supplied:
  - robust pretraining objective: predict close log-return only (1d, easier
    target — avoids the all-4-feats collapse that killed bar_mlm_emb)
  - bigger encoder + warmup/cosine LR schedule (fixes MSE stuck at 0.97)
  - Huber loss for the supervised head (heavy-tailed returns)
  - OOF fine-tuning so train sessions get leakage-free features
  - single scalar feature (not 32d pooled embedding) — less overfit surface
"""
from __future__ import annotations
import math
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
torch.manual_seed(SEED); np.random.seed(SEED)

BARS = 50
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.1
PRETRAIN_EPOCHS = 40
FINETUNE_EPOCHS = 30
BATCH = 128
PRETRAIN_LR = 5e-4
FINETUNE_LR = 3e-4
WARMUP_FRAC = 0.1
MASK_P = 0.15
FWD_HORIZON = 5  # bars 50..54 forward return

# ---------- load sequences ----------
def load_sequences():
    parts = [
        ("tr",   pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")),
        ("pub",  pd.read_parquet(DATA_DIR / "bars_seen_public_test.parquet")),
        ("priv", pd.read_parquet(DATA_DIR / "bars_seen_private_test.parquet")),
    ]
    sids, feats = [], []
    for prefix, df in parts:
        df = df.sort_values(["session", "bar_ix"])
        for sid, g in df.groupby("session", sort=True):
            o = g["open"].to_numpy(np.float64)
            h = g["high"].to_numpy(np.float64)
            l = g["low"].to_numpy(np.float64)
            c = g["close"].to_numpy(np.float64)
            prev = np.concatenate([[c[0]], c[:-1]])
            # 4 features: close-log-ret, log(high/low), (c-l)/(h-l), body/range
            close_ret = np.log(c / prev)
            hl = np.log(np.maximum(h, 1e-12) / np.maximum(l, 1e-12))
            pos_in_range = (c - l) / np.maximum(h - l, 1e-12)
            body_range = (c - o) / np.maximum(h - l, 1e-12)
            f = np.stack([close_ret, hl, pos_in_range, body_range], axis=-1)
            sids.append(f"{prefix}:{int(sid)}")
            feats.append(f.astype(np.float32))
    arr = np.stack(feats, axis=0)
    # sanitize (log of near-zero can produce -inf)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return sids, arr


def load_forward_targets():
    """5-bar forward return for the 1000 train sessions: close[54]/close[49] - 1."""
    unseen = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    unseen = unseen.sort_values(["session", "bar_ix"])
    seen = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    seen = seen.sort_values(["session", "bar_ix"])
    last_seen = seen.groupby("session").tail(1).set_index("session")["close"]
    fwd = unseen[unseen["bar_ix"] == 49 + FWD_HORIZON].set_index("session")["close"]
    y = (fwd / last_seen - 1.0).astype(np.float32)
    return y.sort_index()


print("Loading sequences ...")
sids, seqs = load_sequences()
print(f"  {seqs.shape[0]} sessions, shape {seqs.shape}")
mu = seqs.mean(axis=(0, 1), keepdims=True)
sd = seqs.std(axis=(0, 1), keepdims=True) + 1e-8
seqs_std = ((seqs - mu) / sd).astype(np.float32)

y_fwd = load_forward_targets()
tr_idx = [i for i, s in enumerate(sids) if s.startswith("tr:")]
tr_session_ids = np.array([int(sids[i].split(":")[1]) for i in tr_idx])
y_fwd_arr = y_fwd.reindex(tr_session_ids).to_numpy(np.float32)
print(f"  train-labeled sessions: {len(tr_idx)} (y_fwd mean={y_fwd_arr.mean():+.5f} std={y_fwd_arr.std():.5f})")


# ---------- model ----------
class BarTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(4, D_MODEL)
        self.pos = nn.Embedding(BARS, D_MODEL)
        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=4 * D_MODEL,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.mlm_head = nn.Linear(D_MODEL, 1)  # predict close log-ret only
        self.reg_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.GELU(),
            nn.Dropout(DROPOUT), nn.Linear(D_MODEL, 1),
        )
        self.mask_token = nn.Parameter(torch.zeros(D_MODEL))

    def _trunk(self, x, mask=None):
        h = self.in_proj(x) + self.pos(torch.arange(BARS, device=x.device))
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        return self.enc(h)

    def mlm_forward(self, x, mask):
        h = self._trunk(x, mask)
        return self.mlm_head(h).squeeze(-1)  # (B, 50)

    def predict_fwd(self, x):
        h = self._trunk(x)
        pooled = h.mean(dim=1)
        return self.reg_head(pooled).squeeze(-1)  # (B,)


def cosine_warmup(step, total, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * prog))


# ---------- pretrain ----------
def pretrain(model, X_t, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    N = X_t.shape[0]
    steps_per_epoch = (N + BATCH - 1) // BATCH
    total = steps_per_epoch * epochs
    warmup = int(WARMUP_FRAC * total)
    step = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        losses = []
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            xb = X_t[idx]
            B = xb.shape[0]
            m = (torch.rand(B, BARS, device=DEVICE) < MASK_P)
            if m.sum() == 0:
                continue
            for pg in opt.param_groups:
                pg["lr"] = lr * cosine_warmup(step, total, warmup)
            pred = model.mlm_forward(xb, m)  # (B, 50) — predicts close-logret at all positions
            target = xb[..., 0]  # close-logret channel
            loss = ((pred[m] - target[m]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"  pretrain ep {ep+1:2d}/{epochs}  mlm_mse={np.mean(losses):.5f}")


# ---------- finetune ----------
def finetune(model, X_t, y_t, train_mask, epochs, lr):
    """train_mask: bool tensor over train-labeled sessions."""
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    X_tr = X_t[train_mask]
    y_tr = y_t[train_mask]
    N = X_tr.shape[0]
    steps_per_epoch = (N + BATCH - 1) // BATCH
    total = steps_per_epoch * epochs
    warmup = int(WARMUP_FRAC * total)
    step = 0
    huber = nn.HuberLoss(delta=0.002)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        losses = []
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            xb = X_tr[idx]; yb = y_tr[idx]
            for pg in opt.param_groups:
                pg["lr"] = lr * cosine_warmup(step, total, warmup)
            pred = model.predict_fwd(xb)
            loss = huber(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  finetune ep {ep+1:2d}/{epochs}  huber={np.mean(losses):.6f}")


@torch.no_grad()
def predict_all(model, X_t):
    model.eval()
    out = []
    N = X_t.shape[0]
    for i in range(0, N, BATCH):
        out.append(model.predict_fwd(X_t[i:i + BATCH]).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------- main: OOF fine-tune ----------
X_t = torch.from_numpy(seqs_std).to(DEVICE)
tr_mask_all = torch.zeros(len(sids), dtype=torch.bool, device=DEVICE)
for i in tr_idx:
    tr_mask_all[i] = True

y_fwd_t = torch.zeros(len(sids), dtype=torch.float32, device=DEVICE)
y_fwd_src = torch.from_numpy(y_fwd_arr).to(DEVICE)
idx_t = torch.tensor(tr_idx, dtype=torch.long, device=DEVICE)
y_fwd_t.index_copy_(0, idx_t, y_fwd_src)

N_FOLDS = 5
rng = np.random.default_rng(SEED)
fold_assign = np.tile(np.arange(N_FOLDS), math.ceil(len(tr_idx) / N_FOLDS))[:len(tr_idx)]
rng.shuffle(fold_assign)

oof_pred = np.full(len(tr_idx), np.nan, dtype=np.float32)

print(f"\n=== Pretrain once, fine-tune {N_FOLDS}-fold OOF on {len(tr_idx)} train sessions ===")
print("Pretrain (all 21k sequences) ...")
base_model = BarTransformer().to(DEVICE)
pretrain(base_model, X_t, PRETRAIN_EPOCHS, PRETRAIN_LR)
base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

print(f"\nOOF fine-tune ({N_FOLDS} folds) ...")
for f in range(N_FOLDS):
    model = BarTransformer().to(DEVICE)
    model.load_state_dict(base_state)
    mask = tr_mask_all.clone()
    fold_tr_idx = [tr_idx[i] for i, a in enumerate(fold_assign) if a != f]
    fold_va_idx = [tr_idx[i] for i, a in enumerate(fold_assign) if a == f]
    train_mask = torch.zeros_like(tr_mask_all)
    for i in fold_tr_idx:
        train_mask[i] = True
    print(f"fold {f+1}/{N_FOLDS}: train={len(fold_tr_idx)} val={len(fold_va_idx)}")
    finetune(model, X_t, y_fwd_t, train_mask, FINETUNE_EPOCHS, FINETUNE_LR)
    # predict OOF rows
    model.eval()
    with torch.no_grad():
        va_X = X_t[fold_va_idx]
        va_pred = model.predict_fwd(va_X).cpu().numpy()
    for local_pos, i_global in enumerate(fold_va_idx):
        pos = tr_idx.index(i_global)
        oof_pred[pos] = va_pred[local_pos]

print(f"\nOOF predictions: mean={oof_pred.mean():+.5f} std={oof_pred.std():.5f} "
      f"(target std={y_fwd_arr.std():.5f})")
corr = float(np.corrcoef(oof_pred, y_fwd_arr)[0, 1])
print(f"OOF corr with forward 5-bar return: {corr:+.4f}")

# build per-session feature dataframe
nextmom_tr = pd.Series(oof_pred, index=tr_session_ids, name="nextmom").sort_index()


# ---------- paired-diff ----------
N_SPLITS = 60
HOLDOUT_FRAC = 0.2
ITERS = 52
THRESHOLD_Q = 0.35
HALF = N_SPLITS // 2
N_SEEDS = 3
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

print(f"\nPaired-diff: {N_SPLITS} splits × {N_SEEDS} seeds, baseline vs +nextmom ...")
base_scores = []
mom_scores = []
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
    Xd_m = Xd.assign(nextmom=nextmom_tr.loc[dev_s].to_numpy())
    Xh_m = Xh.assign(nextmom=nextmom_tr.loc[hold_s].to_numpy())
    yd = y_full.loc[dev_s].to_numpy(dtype=float)
    yh = y_full.loc[hold_s].to_numpy(dtype=float)
    vh = np.asarray(Xh["vol"].values, dtype=float)
    p_base = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
    p_mom = np.mean([cb_pred(Xd_m, yd, Xh_m, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
    base_scores.append(score_from_pred(p_base, vh, yh))
    mom_scores.append(score_from_pred(p_mom, vh, yh))
    if (r + 1) % 10 == 0:
        print(f"  scored {r+1}/{N_SPLITS}")

base = np.asarray(base_scores); mom = np.asarray(mom_scores)
d = mom - base
se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean() / se if se > 0 else 0.0
dA, dB = d[:HALF], d[HALF:]
seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean() / seA if seA > 0 else 0.0
seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean() / seB if seB > 0 else 0.0
mark = "*" if dA.mean() > 0 and dB.mean() > 0 else " "

print(f"\nbaseline raw    mean={base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")
print(f"+nextmom        mean={mom.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
      f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")

# save artifacts
out_dir = Path(__file__).resolve().parent.parent / "artifacts"
out_dir.mkdir(exist_ok=True)
nextmom_tr.to_frame().to_parquet(out_dir / "nextmom_train_oof.parquet")
print(f"\nSaved OOF nextmom to {out_dir}/nextmom_train_oof.parquet")
