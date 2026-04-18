"""Build two LB submissions:
  1. catboost_bars_ridge.csv
     Pure RidgeCV on all 40 features + thresholded_inv_vol + finalize(floor=0.40).
     Zero tree-interaction capacity → mechanically bounded overfit.
  2. catboost_bars_seed50_ridge_blend.csv
     0.5 * seed50 CatBoost μ + 0.5 * ridge μ (both z-scored), same shape.
     Splits the bet between tree-interaction signal and linear-only signal.

Paired-diff (60 splits × 3 seeds, seed3 + floor=0.3):
  ridge_all      Δ=-0.014 (half A:-0.001, half B:-0.026) — ~parity with CatBoost

Expected LB: roughly tied with champion 2.586. Upside if CatBoost was
overfitting tree-interactions; downside if those interactions were real signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

from features import (
    SEED, load_train, load_test,
    shape_positions, finalize,
)

ROOT = Path(__file__).resolve().parent.parent

THRESHOLD_Q = 0.35
N_SPLITS = 5
N_CV_REPEATS = 5
N_SEEDS_CB = 50
SHRINK_ALPHA = 0.5
SHORT_FLOOR = 0.40  # match the 2.586 champion

RIDGE_ALPHAS = np.logspace(-3, 3, 13)


X_full, y_full = load_train()
X_test = load_test()
print(f"Train: {X_full.shape}   Test: {X_test.shape}")

# ------ 1. Ridge on all features (deterministic, one fit) ------
print("\n=== Fitting RidgeCV on all 40 features ===")
scaler = StandardScaler()
Xf_std = scaler.fit_transform(X_full.to_numpy(dtype=np.float64))
Xt_std = scaler.transform(X_test.to_numpy(dtype=np.float64))
ridge = RidgeCV(alphas=RIDGE_ALPHAS)
ridge.fit(Xf_std, y_full.to_numpy(dtype=np.float64))
ridge_mu_test = np.asarray(ridge.predict(Xt_std), dtype=float)
print(f"  chosen alpha: {ridge.alpha_:.4g}")
print(f"  ridge μ   mean={ridge_mu_test.mean():+.5f}  std={ridge_mu_test.std():.5f}")

# Build pure-ridge submission
test_vol = np.asarray(X_test["vol"].values, dtype=float)
pos = shape_positions(ridge_mu_test, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
pos = finalize(pos, shrink_alpha=SHRINK_ALPHA, short_floor=SHORT_FLOOR)
sub = pd.DataFrame({"session": X_test.index.astype(int), "target_position": pos})
sub_path_ridge = ROOT / "submissions" / "catboost_bars_ridge.csv"
sub.to_csv(sub_path_ridge, index=False)
print(f"\nSaved: {sub_path_ridge}")
print(sub.describe().loc[["mean", "std", "min", "max"]])

# ------ 2. CatBoost seed50 + ridge blend ------
print("\n=== CV for CatBoost iterations ===")
best_iters: list[int] = []
for repeat in range(N_CV_REPEATS):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat * 97)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full)):
        m = CatBoostRegressor(
            iterations=1500, learning_rate=0.03, depth=5,
            loss_function="MAE", eval_metric="MAE",
            random_seed=SEED + repeat * 97 + fold,
            early_stopping_rounds=100, verbose=False,
        )
        m.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx],
              eval_set=(X_full.iloc[va_idx], y_full.iloc[va_idx]), use_best_model=True)
        best_iters.append(int(m.tree_count_ or 0))
    print(f"  repeat {repeat}: median best_iters so far = {int(np.median(best_iters))}")
final_iters = max(50, int(np.median(best_iters) * (N_SPLITS / (N_SPLITS - 1))))
print(f"FINAL_ITERS = {final_iters}")

print(f"\nFitting {N_SEEDS_CB} CatBoost seeds ...")
cat_preds = []
for k in range(N_SEEDS_CB):
    m = CatBoostRegressor(
        iterations=final_iters, learning_rate=0.03, depth=5,
        loss_function="MAE", random_seed=SEED + k * 1009, verbose=False,
    )
    m.fit(X_full, y_full)
    cat_preds.append(np.asarray(m.predict(X_test), dtype=float))
    if (k + 1) % 10 == 0:
        print(f"  trained {k+1}/{N_SEEDS_CB}")
cat_mu = np.mean(cat_preds, axis=0)
print(f"CatBoost ensemble  mean={cat_mu.mean():+.5f}  std={cat_mu.std():.5f}")

# Z-score both before blending so scales match (ranking preserved — shape is scale-invariant)
cat_z = (cat_mu - cat_mu.mean()) / max(cat_mu.std(), 1e-9)
ridge_z = (ridge_mu_test - ridge_mu_test.mean()) / max(ridge_mu_test.std(), 1e-9)
blend_mu = 0.5 * cat_z + 0.5 * ridge_z
print(f"cat_z/ridge_z corr = {np.corrcoef(cat_z, ridge_z)[0,1]:+.4f}")
print(f"blend μ   mean={blend_mu.mean():+.5f}  std={blend_mu.std():.5f}")

pos = shape_positions(blend_mu, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
pos = finalize(pos, shrink_alpha=SHRINK_ALPHA, short_floor=SHORT_FLOOR)
sub = pd.DataFrame({"session": X_test.index.astype(int), "target_position": pos})
sub_path_blend = ROOT / "submissions" / "catboost_bars_seed50_ridge_blend.csv"
sub.to_csv(sub_path_blend, index=False)
print(f"\nSaved: {sub_path_blend}")
print(sub.describe().loc[["mean", "std", "min", "max"]])
