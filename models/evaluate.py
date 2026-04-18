"""Evaluation for the catboost bars+headlines model.

Runs three diagnostics — all use the same feature builders as catboost_bars.py:
  1. Repeated K-fold CV  -> averaged OOF predictions + RMSE/corr/Sharpe
  2. Frozen holdout       -> bootstrap CI on raw + demeaned Sharpe
  3. Repeated holdout     -> mean ± SE of raw/demeaned Sharpe across splits

The last is the primary metric to chase across feature experiments.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    sharpe, sharpe_bootstrap_ci, shape_positions, finalize,
)

ROOT = Path(__file__).resolve().parent.parent
OOF_DIR = ROOT / "models" / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
N_CV_REPEATS = 5
HOLDOUT_FRAC = 0.2
N_HOLDOUT_REPEATS = 5
BEST_KIND = "thresholded_inv_vol"

# ---------- load ----------
X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()

rng = np.random.default_rng(SEED)
shuffled = rng.permutation(all_sessions)
n_holdout = int(len(shuffled) * HOLDOUT_FRAC)
holdout_sessions = np.sort(shuffled[:n_holdout])
dev_sessions = np.sort(shuffled[n_holdout:])

# Dev gets OOF event features (no leak into dev's own forward returns);
# holdout gets features from impacts fit on the full dev slice.
dev_hdf = headlines_train[headlines_train["session"].isin(dev_sessions)]
dev_bars = bars_train[bars_train["session"].isin(dev_sessions)]
dev_event_oof = build_event_features_oof(dev_hdf, dev_bars, dev_sessions)
frozen_impacts = fit_template_impacts_multi(dev_hdf, dev_bars)
hold_hdf = headlines_train[headlines_train["session"].isin(holdout_sessions)]
hold_event = build_event_features_multi(hold_hdf, holdout_sessions, frozen_impacts)

X = X_base.loc[dev_sessions].join(dev_event_oof)
X_holdout = X_base.loc[holdout_sessions].join(hold_event)
y = y_full.loc[dev_sessions]
y_holdout = y_full.loc[holdout_sessions]
X_full = pd.concat([X, X_holdout]).sort_index()
print(f"Features: {X_full.shape[1]} cols")
print(f"Dev: {len(X)} sessions | Holdout: {len(X_holdout)} sessions")
print(f"Dev y: mean={y.mean():.5f} std={y.std():.5f}")
print(f"Holdout y: mean={y_holdout.mean():.5f} std={y_holdout.std():.5f}")


# ---------- 1. repeated K-fold CV ----------
oof_accum = np.zeros(len(X))
best_iters: list[int] = []
for repeat in range(N_CV_REPEATS):
    oof_r = np.zeros(len(X))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat * 97)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        m = CatBoostRegressor(
            iterations=1500, learning_rate=0.03, depth=5,
            loss_function="RMSE", eval_metric="RMSE",
            random_seed=SEED + repeat * 97 + fold,
            early_stopping_rounds=100, verbose=False,
        )
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=(X.iloc[va_idx], y.iloc[va_idx]), use_best_model=True)
        oof_r[va_idx] = m.predict(X.iloc[va_idx])
        best_iters.append(m.tree_count_)
    oof_accum += oof_r
    print(f"  repeat {repeat}: OOF rmse={np.sqrt(((oof_r - np.asarray(y.values, dtype=float)) ** 2).mean()):.5f}")

oof = oof_accum / N_CV_REPEATS
y_arr = np.asarray(y.values, dtype=float)
print(f"\nAveraged OOF ({N_CV_REPEATS}x{N_SPLITS}-fold) RMSE: {np.sqrt(np.mean((oof - y_arr) ** 2)):.5f}")
print(f"OOF corr(pred, y): {np.corrcoef(oof, y_arr)[0, 1]:.4f}")

oof_vol = np.asarray(X["vol"].values, dtype=float)
print("\nProxy Sharpe on OOF predictions, by position-shaping rule (FYI):")
for kind in ["raw", "sign", "inv_vol", "thresholded", "thresholded_inv_vol"]:
    print(f"  {kind:>22}: Sharpe = {sharpe(shape_positions(oof, oof_vol, kind), y_arr):.4f}")
print(f"Using rule: {BEST_KIND!r}")

out = pd.DataFrame({"session": X.index, "y_true": y.values, "y_pred": oof})
out.to_parquet(OOF_DIR / "catboost_bars.parquet", index=False)
print(f"Saved OOF: {OOF_DIR / 'catboost_bars.parquet'}")


# ---------- 2. frozen holdout ----------
print("\n--- Frozen holdout (dev-trained, holdout-scored) ---")
honest_iters = max(50, int(np.median(best_iters) * (N_SPLITS / (N_SPLITS - 1))))
dev_model = CatBoostRegressor(
    iterations=honest_iters, learning_rate=0.03, depth=5,
    loss_function="RMSE", random_seed=SEED, verbose=False,
)
dev_model.fit(X, y)
hold_pred = np.asarray(dev_model.predict(X_holdout), dtype=float)
hold_vol = np.asarray(X_holdout["vol"].values, dtype=float)
hold_y = np.asarray(y_holdout.values, dtype=float)
hold_pos = shape_positions(hold_pred, hold_vol, BEST_KIND)

raw_s, raw_lo, raw_hi = sharpe_bootstrap_ci(hold_pos, hold_y)
print(f"  HOLDOUT raw Sharpe   = {raw_s:.3f}  [95% CI {raw_lo:.2f}, {raw_hi:.2f}]")
print(f"  HOLDOUT corr(pred,y) = {np.corrcoef(hold_pred, hold_y)[0, 1]:+.4f}")

oof_dm_s, oof_dm_lo, oof_dm_hi = sharpe_bootstrap_ci(
    shape_positions(oof - oof.mean(), oof_vol, BEST_KIND), y_arr)
hold_dm_s, hold_dm_lo, hold_dm_hi = sharpe_bootstrap_ci(
    shape_positions(hold_pred - hold_pred.mean(), hold_vol, BEST_KIND), hold_y)
print("\n=== PRIMARY METRIC: demeaned Sharpe (pure cross-sectional skill) ===")
print(f"  OOF     demeaned = {oof_dm_s:+.3f}  [95% CI {oof_dm_lo:+.2f}, {oof_dm_hi:+.2f}]")
print(f"  HOLDOUT demeaned = {hold_dm_s:+.3f}  [95% CI {hold_dm_lo:+.2f}, {hold_dm_hi:+.2f}]")

c_oof, c_oof_lo, c_oof_hi = sharpe_bootstrap_ci(np.ones_like(y_arr), y_arr)
c_h, c_h_lo, c_h_hi = sharpe_bootstrap_ci(np.ones_like(hold_y), hold_y)
print("\nBaselines (constant 'always long'):")
print(f"  OOF     = {c_oof:.3f}  [95% CI {c_oof_lo:.2f}, {c_oof_hi:.2f}]")
print(f"  HOLDOUT = {c_h:.3f}  [95% CI {c_h_lo:.2f}, {c_h_hi:.2f}]")


# ---------- 3. repeated holdout ----------
print(f"\n--- Repeated holdout: {N_HOLDOUT_REPEATS} independent 80/20 splits ---")
rh_raw, rh_demeaned, rh_corr = [], [], []
rh_rng = np.random.default_rng(SEED + 1)
for r in range(N_HOLDOUT_REPEATS):
    sh = rh_rng.permutation(all_sessions)
    hold_s = np.sort(sh[:n_holdout])
    dev_s = np.sort(sh[n_holdout:])
    # OOF event features on dev; dev-fitted impacts applied to holdout.
    dev_h = headlines_train[headlines_train["session"].isin(dev_s)]
    hold_h = headlines_train[headlines_train["session"].isin(hold_s)]
    dev_b = bars_train[bars_train["session"].isin(dev_s)]
    dev_event = build_event_features_oof(dev_h, dev_b, dev_s)
    split_impacts = fit_template_impacts_multi(dev_h, dev_b)
    hold_event = build_event_features_multi(hold_h, hold_s, split_impacts)
    Xd = X_base.loc[dev_s].join(dev_event)
    Xh = X_base.loc[hold_s].join(hold_event)
    yd, yh = y_full.loc[dev_s], y_full.loc[hold_s]
    m = CatBoostRegressor(
        iterations=honest_iters, learning_rate=0.03, depth=5,
        loss_function="RMSE", random_seed=SEED + r, verbose=False,
    )
    m.fit(Xd, yd)
    p = np.asarray(m.predict(Xh), dtype=float)
    vh = np.asarray(Xh["vol"].values, dtype=float)
    yh_arr = np.asarray(yh.values, dtype=float)
    raw_pos = shape_positions(p, vh, BEST_KIND)
    final_pos = finalize(raw_pos)
    rh_raw.append(sharpe(raw_pos, yh_arr))
    rh_demeaned.append(sharpe(shape_positions(p - p.mean(), vh, BEST_KIND), yh_arr))
    rh_corr.append(float(np.corrcoef(p, yh_arr)[0, 1]))
    print(f"  split {r}: raw={rh_raw[-1]:+.3f}  final={sharpe(final_pos, yh_arr):+.3f}  "
          f"const={sharpe(np.ones_like(yh_arr), yh_arr):+.3f}  demeaned={rh_demeaned[-1]:+.3f}")

rh_raw_arr = np.array(rh_raw)
rh_dm_arr = np.array(rh_demeaned)
print(f"\n=== REPEATED-HOLDOUT SUMMARY ({N_HOLDOUT_REPEATS} splits) ===")
print(f"  Mean raw Sharpe      = {rh_raw_arr.mean():+.3f} ± {rh_raw_arr.std(ddof=1) / np.sqrt(N_HOLDOUT_REPEATS):.3f}  (1 SE)")
print(f"  Mean demeaned Sharpe = {rh_dm_arr.mean():+.3f} ± {rh_dm_arr.std(ddof=1) / np.sqrt(N_HOLDOUT_REPEATS):.3f}  (1 SE)  <-- PRIMARY")
print(f"  Mean corr(pred,y)    = {np.mean(rh_corr):+.4f}")
