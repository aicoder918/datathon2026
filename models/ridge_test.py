"""Paired-diff: radical capacity reduction via ridge / elastic-net regression.

Hypothesis: CatBoost can represent millions of tree interactions on 1k samples,
which is precisely the capacity that fits 1k train idiosyncrasies and fails to
transfer (8 strikes on LB). A linear model with L2 or L1+L2 regularization and
<=10 features literally cannot represent interactions — capacity is
mechanically bounded, so whatever it learns must be surface-level trends that
should generalize.

Variants:
  baseline       — seed3 CatBoost + thresholded_inv_vol + finalize
  ridge_all      — RidgeCV on ALL 40 features (no selection)
  ridge_top10    — RidgeCV on top-10 |corr(x, y)| features
  ridge_top5     — RidgeCV on top-5 |corr(x, y)| features
  enet_top10     — ElasticNetCV on top-10 features

All linear variants are z-scored on dev stats, use the SAME shape pipeline
(thresholded_inv_vol + finalize) as baseline — so the only thing changing is
the prediction model. Linear models are deterministic so no seed averaging.

Expected by refined 8-strike rule: linear models don't introduce a new
per-sample signal (same features in, same shape pipeline out). If ANY untried
lever can transfer, this is the one — but the local Δ is likely strongly
negative because CatBoost's expressive power was real on 1k training. The
question is whether local loss matches LB loss or whether linear's
better-generalization reverses the sign.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
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

CB_KW = dict(iterations=ITERS, learning_rate=0.03, depth=5,
             loss_function="MAE", verbose=False)

RIDGE_ALPHAS = np.logspace(-3, 3, 13)
ENET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]


def cb_pred(Xd, yd, Xh, seed):
    m = CatBoostRegressor(**CB_KW, random_seed=seed)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


def select_topk(Xd: np.ndarray, yd: np.ndarray, k: int) -> np.ndarray:
    """Return column indices of top-k features by |Pearson r| with yd, computed on dev."""
    corrs = np.array([
        abs(np.corrcoef(Xd[:, j], yd)[0, 1]) if np.std(Xd[:, j]) > 1e-12 else 0.0
        for j in range(Xd.shape[1])
    ])
    return np.argsort(-corrs)[:k]


def linear_pred(Xd_np: np.ndarray, yd_np: np.ndarray, Xh_np: np.ndarray,
                model_name: str, feat_idx: np.ndarray | None = None) -> np.ndarray:
    """Fit a linear model on (selected columns of) Xd and predict on Xh."""
    if feat_idx is not None:
        Xd_np = Xd_np[:, feat_idx]
        Xh_np = Xh_np[:, feat_idx]
    scaler = StandardScaler()
    Xd_std = scaler.fit_transform(Xd_np)
    Xh_std = scaler.transform(Xh_np)
    if model_name == "ridge":
        m = RidgeCV(alphas=RIDGE_ALPHAS)
    elif model_name == "enet":
        m = ElasticNetCV(l1_ratio=ENET_L1_RATIOS, alphas=None, cv=3,
                         max_iter=5000, n_jobs=1, random_state=SEED)
    else:
        raise ValueError(model_name)
    m.fit(Xd_std, yd_np)
    return np.asarray(m.predict(Xh_std), dtype=float)


def score_from_pred(pred, vol_h, y_h):
    pos = shape_positions(pred, vol_h, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    return sharpe(pos, y_h)


X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)
rng = np.random.default_rng(SEED + 1)
splits = [tuple(np.sort(s) for s in (sh[:n_holdout], sh[n_holdout:]))
          for sh in (rng.permutation(all_sessions) for _ in range(N_SPLITS))]

VARIANTS = ["baseline", "ridge_all", "ridge_top10", "ridge_top5", "enet_top10"]
scores = {v: [] for v in VARIANTS}

print(f"Paired-diff: {N_SPLITS} splits × {N_SEEDS_CB} Cat seeds ...")
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

    # baseline — CatBoost seed3
    mus = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS_CB)], axis=0)
    scores["baseline"].append(score_from_pred(mus, vh, yh))

    Xd_np = Xd.to_numpy(dtype=np.float64)
    Xh_np = Xh.to_numpy(dtype=np.float64)

    # ridge on all features
    p = linear_pred(Xd_np, yd, Xh_np, "ridge", feat_idx=None)
    scores["ridge_all"].append(score_from_pred(p, vh, yh))

    # ridge on top-10 (feature selection on dev only)
    idx10 = select_topk(Xd_np, yd, 10)
    p = linear_pred(Xd_np, yd, Xh_np, "ridge", feat_idx=idx10)
    scores["ridge_top10"].append(score_from_pred(p, vh, yh))

    # ridge on top-5
    idx5 = select_topk(Xd_np, yd, 5)
    p = linear_pred(Xd_np, yd, Xh_np, "ridge", feat_idx=idx5)
    scores["ridge_top5"].append(score_from_pred(p, vh, yh))

    # enet on top-10
    p = linear_pred(Xd_np, yd, Xh_np, "enet", feat_idx=idx10)
    scores["enet_top10"].append(score_from_pred(p, vh, yh))

    if (r + 1) % 10 == 0:
        top5_names = list(Xd.columns[idx5])
        print(f"  scored {r+1}/{N_SPLITS}   top5 last split: {top5_names}")

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
    print(f"  {v:<14} mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
          f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")

print("\nNote: if any ridge variant comes CLOSE to baseline (|Δ| < 0.2) with")
print("both halves aligned, it's worth an LB slot — linear models have bounded")
print("overfit and could outperform their local Sharpe at 20k scale even if the")
print("local Δ is negative. Catastrophic Δ (< -0.3) means linear really can't")
print("capture enough signal and LB would also be worse.")
