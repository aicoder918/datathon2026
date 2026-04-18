"""Compare CatBoost loss functions on the repeated-holdout benchmark.

RMSE chases outliers; returns are heavy-tailed, so MAE/Huber might help.
Uses the same splits across all losses so the comparison is fair.
"""
from __future__ import annotations
import numpy as np
from catboost import CatBoostRegressor
from features import (
    SEED, load_train_base,
    fit_template_impacts_multi, build_event_features_multi, build_event_features_oof,
    sharpe, shape_positions,
)

N_SPLITS_HOLDOUT = 40
HOLDOUT_FRAC = 0.2
ITERS = 52  # matches catboost_bars.py FINAL_ITERS
BEST_KIND = "thresholded_inv_vol"

LOSSES = [
    ("RMSE", "RMSE"),
    ("MAE",  "MAE"),
    ("Huber:delta=0.01", "Huber δ=0.01"),
    ("Huber:delta=0.02", "Huber δ=0.02"),
    ("Huber:delta=0.05", "Huber δ=0.05"),
]

X_base, y_full, headlines_train, bars_train = load_train_base()
all_sessions = X_base.index.to_numpy()
n_holdout = int(len(all_sessions) * HOLDOUT_FRAC)

rng = np.random.default_rng(SEED + 1)
splits = []
for _ in range(N_SPLITS_HOLDOUT):
    sh = rng.permutation(all_sessions)
    splits.append((np.sort(sh[:n_holdout]), np.sort(sh[n_holdout:])))

# Pre-build features per split (same across all losses).
print(f"Target std: {y_full.std():.5f}  (for Huber delta context)")
print("Building features per split ...")
prepped = []
for hold_s, dev_s in splits:
    dev_h = headlines_train[headlines_train["session"].isin(dev_s)]
    hold_h = headlines_train[headlines_train["session"].isin(hold_s)]
    dev_b = bars_train[bars_train["session"].isin(dev_s)]
    dev_event = build_event_features_oof(dev_h, dev_b, dev_s)
    split_impacts = fit_template_impacts_multi(dev_h, dev_b)
    hold_event = build_event_features_multi(hold_h, hold_s, split_impacts)
    Xd = X_base.loc[dev_s].join(dev_event)
    Xh = X_base.loc[hold_s].join(hold_event)
    yd, yh = y_full.loc[dev_s], y_full.loc[hold_s]
    prepped.append((Xd, yd, Xh, yh))

print()
results = {}
for loss_arg, label in LOSSES:
    rh_raw, rh_dm = [], []
    for r, (Xd, yd, Xh, yh) in enumerate(prepped):
        m = CatBoostRegressor(
            iterations=ITERS, learning_rate=0.03, depth=5,
            loss_function=loss_arg, random_seed=SEED + r, verbose=False,
        )
        m.fit(Xd, yd)
        p = np.asarray(m.predict(Xh), dtype=float)
        vh = np.asarray(Xh["vol"].values, dtype=float)
        yh_arr = np.asarray(yh.values, dtype=float)
        rh_raw.append(sharpe(shape_positions(p, vh, BEST_KIND), yh_arr))
        rh_dm.append(sharpe(shape_positions(p - p.mean(), vh, BEST_KIND), yh_arr))
    raw = np.array(rh_raw); dm = np.array(rh_dm)
    results[label] = (dm, raw)
    print(f"  {label:>16}: demeaned = {dm.mean():+.3f} ± {dm.std(ddof=1)/np.sqrt(len(dm)):.3f}   "
          f"raw = {raw.mean():+.3f} ± {raw.std(ddof=1)/np.sqrt(len(raw)):.3f}")

print("\n=== RANKED BY DEMEANED SHARPE ===")
for label, (dm, raw) in sorted(results.items(), key=lambda kv: -kv[1][0].mean()):
    print(f"  {label:>16}: demeaned {dm.mean():+.3f}   raw {raw.mean():+.3f}")

# ---------- paired differences vs RMSE ----------
# Same splits across losses → split-level noise cancels, so paired SE
# is much tighter than the marginal SEs above.
print("\n=== PAIRED DIFF vs RMSE (same splits, so split noise cancels) ===")
rmse_dm, rmse_raw = results["RMSE"]
n = len(rmse_dm)
for label, (dm, raw) in results.items():
    if label == "RMSE":
        continue
    d_dm = dm - rmse_dm
    d_raw = raw - rmse_raw
    se_dm = d_dm.std(ddof=1) / np.sqrt(n)
    se_raw = d_raw.std(ddof=1) / np.sqrt(n)
    t_dm = d_dm.mean() / se_dm if se_dm > 0 else 0.0
    t_raw = d_raw.mean() / se_raw if se_raw > 0 else 0.0
    print(f"  {label:>16}: Δdemeaned = {d_dm.mean():+.3f} ± {se_dm:.3f}  (t={t_dm:+.2f})   "
          f"Δraw = {d_raw.mean():+.3f} ± {se_raw:.3f}  (t={t_raw:+.2f})")
