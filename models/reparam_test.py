"""Paired-diff: pure re-parameterization of existing features (same info, new
coordinates). No new signal — just log/scale transforms of columns the tree
already sees. Tests whether the class of change that *doesn't add signal*
avoids the 1k→20k transfer failure.

Variants (all REPLACE, not add):
  log1p_counts   — hl_n, hl_n_recent, hl_n_pos, hl_n_neg → log1p(col)
  log1p_all_nn   — also: abs-then-log1p on max_drawdown (non-positive),
                   vol×10000, vol_recent×10000 (tiny-scale → logs)
  asinh_ret      — seen_ret, mom_1..10 → asinh(x × 100) (symmetric log-like,
                   handles negatives; compresses tails)
  all            — union of above (but applied to disjoint col sets)

Baseline = unchanged features. Expectation: if reparam transfers where new
features don't, Δ should be small-positive or near-zero with consistent halves
and no active hurt.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    fit_template_impacts_sector_multi, build_event_features_sector_multi,
    build_event_features_sector_oof,
    sharpe, shape_positions,
)

N_SPLITS = 60
HOLDOUT_FRAC = 0.2
ITERS = 52
THRESHOLD_Q = 0.35
HALF = N_SPLITS // 2
N_SEEDS = 3
CB_KW = dict(iterations=ITERS, learning_rate=0.03, depth=5,
             loss_function="MAE", verbose=False)

LOG1P_COUNT_COLS = ["hl_n", "hl_n_recent", "hl_n_pos", "hl_n_neg"]
ASINH_RET_COLS   = ["seen_ret", "mom_1", "mom_3", "mom_5", "mom_10"]
OTHER_NN_COLS    = ["vol", "vol_recent"]   # tiny-scale positive


def reparam(X: pd.DataFrame, variant: str) -> pd.DataFrame:
    X = X.copy()
    if variant in ("log1p_counts", "log1p_all_nn", "all"):
        for c in LOG1P_COUNT_COLS:
            if c in X.columns:
                X[c] = np.log1p(X[c].clip(lower=0))
    if variant in ("log1p_all_nn", "all"):
        for c in OTHER_NN_COLS:
            if c in X.columns:
                X[c] = np.log1p(X[c].clip(lower=0) * 1e4)
        if "max_drawdown" in X.columns:
            # max_drawdown ∈ (-∞, 0]; reparam as -log1p(-x)
            X["max_drawdown"] = -np.log1p(-X["max_drawdown"].clip(upper=0))
    if variant in ("asinh_ret", "all"):
        for c in ASINH_RET_COLS:
            if c in X.columns:
                X[c] = np.arcsinh(X[c].to_numpy() * 100.0)
    return X


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

VARIANTS = ["baseline", "log1p_counts", "log1p_all_nn", "asinh_ret", "all"]
scores = {v: [] for v in VARIANTS}

print(f"Paired-diff: {N_SPLITS} splits × {N_SEEDS} seeds, {len(VARIANTS)} variants ...")
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
    Xd_full = X_base.loc[dev_s].join(dev_event).join(dev_sec)
    Xh_full = X_base.loc[hold_s].join(hold_event).join(hold_sec)
    yd = y_full.loc[dev_s].to_numpy(dtype=float)
    yh = y_full.loc[hold_s].to_numpy(dtype=float)
    vh = np.asarray(Xh_full["vol"].values, dtype=float)

    for v in VARIANTS:
        Xd = Xd_full if v == "baseline" else reparam(Xd_full, v)
        Xh = Xh_full if v == "baseline" else reparam(Xh_full, v)
        pr = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
        scores[v].append(score_from_pred(pr, vh, yh))
    if (r + 1) % 10 == 0:
        print(f"  scored {r+1}/{N_SPLITS}")

base = np.asarray(scores["baseline"])
print(f"\nbaseline raw  mean={base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")
print("\nReparam variants:")
for v in VARIANTS[1:]:
    s = np.asarray(scores[v])
    d = s - base
    se = d.std(ddof=1) / np.sqrt(N_SPLITS); t = d.mean() / se if se > 0 else 0.0
    dA, dB = d[:HALF], d[HALF:]
    seA = dA.std(ddof=1) / np.sqrt(HALF); tA = dA.mean() / seA if seA > 0 else 0.0
    seB = dB.std(ddof=1) / np.sqrt(HALF); tB = dB.mean() / seB if seB > 0 else 0.0
    mark = "*" if dA.mean() > 0 and dB.mean() > 0 else " "
    print(f"  {v:<14} mean={s.mean():+.3f}  Δ={d.mean():+.3f}±{se:.3f}(t={t:+.2f})  "
          f"A:{dA.mean():+.3f}(t={tA:+.2f}) B:{dB.mean():+.3f}(t={tB:+.2f}) {mark}")

print("\nSanity check: if CatBoost is split-invariant to monotone transforms of "
      "continuous features, ALL Δ should be ≈0 (within split noise). Any non-null "
      "result means trees aren't fully invariant here — likely because the transforms "
      "change how ties/binning work in CatBoost's quantization.")
