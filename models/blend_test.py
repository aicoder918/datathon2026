"""Paired-diff: cross-family blend of CatBoost + LightGBM. Same features, same
target, same shape pipeline. Only the MODEL changes.

Rationale: seed ensembling (same family, different seeds) transferred +0.013 to
LB. Cross-family blend should reduce variance more because different algos make
different systematic errors. This is the only untested lever in the proven
variance-reduction class.

Variants:
  baseline   — 3-seed CatBoost (current submission's class, N_SEEDS=3 for speed)
  lgb3       — 3-seed LightGBM alone (feasibility check)
  cat3+lgb3  — 50/50 prediction average
  cat3+lgb3_w — inverse-CV-MAE weighted blend (family with lower dev MAE gets
                higher weight per split)

Bar for LB submission: Δ > 0 on BOTH halves AND Δ modest (< 0.10). Seed ensemble
got +0.013 locally → +0.013 LB. Anything much bigger locally is the 7-strike
trap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb

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
LGB_KW = dict(
    n_estimators=ITERS, learning_rate=0.03, num_leaves=31,
    max_depth=5, objective="regression_l1", min_data_in_leaf=20,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
    verbosity=-1,
)


def cb_pred(Xd, yd, Xh, seed):
    m = CatBoostRegressor(**CB_KW, random_seed=seed)
    m.fit(Xd, yd)
    return np.asarray(m.predict(Xh), dtype=float)


def lgb_pred(Xd, yd, Xh, seed):
    m = lgb.LGBMRegressor(**LGB_KW, random_state=seed, seed=seed)
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

VARIANTS = ["baseline", "lgb3", "cat3+lgb3", "cat3+lgb3_w"]
scores = {v: [] for v in VARIANTS}

print(f"Paired-diff: {N_SPLITS} splits × {N_SEEDS} seeds ...")
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

    # Seed-averaged predictions for each family
    cat_preds = np.mean([cb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)
    lgb_preds = np.mean([lgb_pred(Xd, yd, Xh, seed=SEED + r * 997 + k) for k in range(N_SEEDS)], axis=0)

    # For weighted blend: compute each family's dev MAE via quick holdout inside dev
    # Use one seed per family for speed; weight inverse to MAE.
    n_dev = len(dev_s)
    n_val = n_dev // 5
    val_mask = np.arange(n_dev) < n_val  # first 20% of dev sessions
    Xd_sub = Xd.iloc[~val_mask]; yd_sub = yd[~val_mask]
    Xd_val = Xd.iloc[val_mask];  yd_val = yd[val_mask]
    cat_val_pred = cb_pred(Xd_sub, yd_sub, Xd_val, seed=SEED + r * 997)
    lgb_val_pred = lgb_pred(Xd_sub, yd_sub, Xd_val, seed=SEED + r * 997)
    cat_mae = float(np.mean(np.abs(cat_val_pred - yd_val)))
    lgb_mae = float(np.mean(np.abs(lgb_val_pred - yd_val)))
    # Inverse-MAE weights, normalized
    w_cat = 1.0 / max(cat_mae, 1e-9)
    w_lgb = 1.0 / max(lgb_mae, 1e-9)
    w_sum = w_cat + w_lgb
    w_cat /= w_sum; w_lgb /= w_sum

    scores["baseline"].append(score_from_pred(cat_preds, vh, yh))
    scores["lgb3"].append(score_from_pred(lgb_preds, vh, yh))
    scores["cat3+lgb3"].append(score_from_pred(0.5 * cat_preds + 0.5 * lgb_preds, vh, yh))
    scores["cat3+lgb3_w"].append(score_from_pred(w_cat * cat_preds + w_lgb * lgb_preds, vh, yh))

    if (r + 1) % 10 == 0:
        print(f"  scored {r+1}/{N_SPLITS}   last weights cat={w_cat:.2f} lgb={w_lgb:.2f}")

base = np.asarray(scores["baseline"])
print(f"\nbaseline (cat3) raw  mean={base.mean():+.3f} ± {base.std(ddof=1)/np.sqrt(N_SPLITS):.3f}")
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

print("\nBar for LB submission:")
print("  * both halves positive AND Δ modest (< 0.10) — aligns with variance-reduction class")
print("  Δ big (> 0.12) → likely the 7-strike trap, DO NOT submit")
print("  lgb3 alone much worse than baseline → family mismatch, blend won't help")
