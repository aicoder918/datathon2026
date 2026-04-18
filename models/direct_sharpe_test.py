"""Paired-diff: direct Sharpe maximization. Small MLP on the current feature
set, output = raw tilt via tanh, loss = differentiable batched Sharpe on dev.

Rationale: the predict-then-shape pipeline has two degrees of freedom
(prediction + hand-designed shaping) we never jointly optimize. A neural net
trained directly against the scoring rule should either find a better mapping
or reveal that the current heuristic is already near-optimal.

Variants:
  baseline          — current: seed3 CatBoost + thresholded_inv_vol + finalize
  mlp_raw           — MLP(tanh) → positions directly, no finalize
  mlp_finalized     — MLP(tanh) → finalize (shrink-to-long + floor shorts at 0.3)
  mlp_linear        — SINGLE linear layer + tanh (zero hidden layers)
  mlp_linear_fin    — linear + tanh + finalize

Finalize is a proven-transfer lever (+0.006 on LB). Testing both raw and
finalized variants isolates whether gains come from smarter sizing or from
interacting with the kill-shorts rule.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor

from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    fit_template_impacts_sector_multi, build_event_features_sector_multi,
    build_event_features_sector_oof,
    sharpe, shape_positions, finalize,
)

N_SPLITS = 60
HOLDOUT_FRAC = 0.2
ITERS = 52
THRESHOLD_Q = 0.35
HALF = N_SPLITS // 2
N_SEEDS_CB = 3

DEVICE = torch.device("cpu")  # 30 features x 800 rows — CPU is fine and avoids MPS overhead
EPOCHS = 300
LR = 3e-3
WD = 1e-2
HIDDEN = 32
DROPOUT = 0.2
N_SEEDS_MLP = 3   # seed-average MLP positions too (same variance-reduction rationale)

CB_KW = dict(iterations=ITERS, learning_rate=0.03, depth=5,
             loss_function="MAE", verbose=False)


def cb_pred(Xd, yd, Xh, seed):
    m = CatBoostRegressor(**CB_KW, random_seed=seed)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


class DirectSharpeMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return torch.tanh(self.net(x)).squeeze(-1)


class DirectSharpeLinear(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lin = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.tanh(self.lin(x)).squeeze(-1)


def sharpe_loss(pos: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative annualized Sharpe on positions·returns."""
    pnl = pos * y
    mean = pnl.mean()
    std = pnl.std() + 1e-8
    return -(mean / std) * 16.0


def train_one_seed(Xd_np: np.ndarray, yd_np: np.ndarray, Xh_np: np.ndarray,
                   seed: int, use_linear: bool = False) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # standardize by dev stats
    mu = Xd_np.mean(0)
    sd = Xd_np.std(0).clip(1e-6)
    Xd = torch.tensor((Xd_np - mu) / sd, dtype=torch.float32, device=DEVICE)
    Xh = torch.tensor((Xh_np - mu) / sd, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(yd_np, dtype=torch.float32, device=DEVICE)

    n_features = Xd.shape[1]
    if use_linear:
        model = DirectSharpeLinear(n_features).to(DEVICE)
    else:
        model = DirectSharpeMLP(n_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # use every-epoch train-sharpe as the optimization target; no ES because
    # Sharpe on 800 rows is noisy at sub-epoch granularity and we want the
    # fixed-compute variant for clean paired-diff.
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pos = model(Xd)
        loss = sharpe_loss(pos, y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xh).cpu().numpy()


def direct_sharpe_pos(Xd_np, yd_np, Xh_np, base_seed: int, use_linear: bool = False) -> np.ndarray:
    preds = []
    for k in range(N_SEEDS_MLP):
        pos = train_one_seed(Xd_np, yd_np, Xh_np, seed=base_seed + k * 1009, use_linear=use_linear)
        preds.append(pos)
    return np.mean(preds, axis=0)


def sharpe_score(pos, y):
    return sharpe(pos, y)


X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

VARIANTS = ["baseline", "mlp_raw", "mlp_finalized", "mlp_linear", "mlp_linear_fin"]
scores = {v: [] for v in VARIANTS}

print(f"Paired-diff: {N_SPLITS} splits × {N_SEEDS_MLP} MLP seeds / {N_SEEDS_CB} Cat seeds ...")
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
    yd = y_full.loc[dev_s].to_numpy(dtype=float)
    yh = y_full.loc[hold_s].to_numpy(dtype=float)
    vh = np.asarray(Xh["vol"].values, dtype=float)

    # baseline — seed3 CatBoost + thresholded_inv_vol + finalize
    mus = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS_CB)], axis=0)
    pos_base = shape_positions(mus, vh, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos_base = finalize(pos_base)
    scores["baseline"].append(sharpe_score(pos_base, yh))

    Xd_np = Xd.to_numpy(dtype=np.float32)
    Xh_np = Xh.to_numpy(dtype=np.float32)

    # MLP (hidden=32)
    mlp_pos = direct_sharpe_pos(Xd_np, yd.astype(np.float32), Xh_np,
                                 base_seed=SEED + r * 997, use_linear=False)
    scores["mlp_raw"].append(sharpe_score(mlp_pos, yh))
    scores["mlp_finalized"].append(sharpe_score(finalize(mlp_pos), yh))

    # Linear
    lin_pos = direct_sharpe_pos(Xd_np, yd.astype(np.float32), Xh_np,
                                 base_seed=SEED + r * 997, use_linear=True)
    scores["mlp_linear"].append(sharpe_score(lin_pos, yh))
    scores["mlp_linear_fin"].append(sharpe_score(finalize(lin_pos), yh))

    if (r + 1) % 5 == 0:
        print(f"  scored {r+1}/{N_SPLITS}   "
              f"mlp|mlp_fin|lin|lin_fin={scores['mlp_raw'][-1]:+.2f}|{scores['mlp_finalized'][-1]:+.2f}|"
              f"{scores['mlp_linear'][-1]:+.2f}|{scores['mlp_linear_fin'][-1]:+.2f}  "
              f"base={scores['baseline'][-1]:+.2f}")

base = np.asarray(scores["baseline"])
print(f"\nbaseline raw  mean={base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")
for v in VARIANTS[1:]:
    s = np.asarray(scores[v])
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean() / se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean() / seA if seA > 0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean() / seB if seB > 0 else 0.0
    mark = "*" if dA.mean() > 0 and dB.mean() > 0 else (
           "!" if dA.mean() < 0 and dB.mean() < 0 else " ")
    print(f"  {v:<17} mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
          f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")
