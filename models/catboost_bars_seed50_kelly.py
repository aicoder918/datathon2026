"""Kelly-shape on top of the seed50 + short_floor=0.40 base.

Stacks three orthogonal variance-reduction levers:
  1. 50 seeds of MAE CatBoost → low-variance μ predictions
  2. Kelly sizing: position ∝ μ / (vol² + σ²_model), replacing thresholded_inv_vol
     where σ²_model comes from averaged RMSEWithUncertainty virtual ensembles
  3. finalize with short_floor=0.40 (matching the 2.586 champion)

Paired-diff on seed3 + floor=0.30 showed kelly_combined Δ=+0.026 (t=+2.84, both
halves positive). Mechanisms are independent of seed count and floor height so
gains should compose additively with the 2.586 base.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

from features import (
    SEED, load_train, load_test,
    finalize,
)

ROOT = Path(__file__).resolve().parent.parent

THRESHOLD_Q = 0.35
N_SPLITS = 5
N_CV_REPEATS = 5
N_SEEDS_MU = 50
N_SEEDS_SIGMA = 5
SHRINK_ALPHA = 0.5
SHORT_FLOOR = 0.40  # match the 2.586 champion, NOT the default 0.30


X_full, y_full = load_train()
X_test = load_test()
print(f"Train: {X_full.shape}   Test: {X_test.shape}")

print(f"\nRunning {N_CV_REPEATS}×{N_SPLITS}-fold CV to pick iterations...")
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

print(f"\nFitting {N_SEEDS_MU} seed-diversified MAE CatBoosts for μ ...")
mu_preds = []
for k in range(N_SEEDS_MU):
    m = CatBoostRegressor(
        iterations=final_iters, learning_rate=0.03, depth=5,
        loss_function="MAE", random_seed=SEED + k * 1009, verbose=False,
    )
    m.fit(X_full, y_full)
    mu_preds.append(np.asarray(m.predict(X_test), dtype=float))
    if (k + 1) % 10 == 0:
        print(f"  trained {k+1}/{N_SEEDS_MU} μ-models")
mu = np.mean(mu_preds, axis=0)
print(f"μ ensemble  mean={mu.mean():+.5f}  std={mu.std():.5f}")

print(f"\nFitting {N_SEEDS_SIGMA} RMSEWithUncertainty CatBoosts for σ² ...")
var_preds = []
for k in range(N_SEEDS_SIGMA):
    m = CatBoostRegressor(
        iterations=final_iters, learning_rate=0.03, depth=5,
        loss_function="RMSEWithUncertainty",
        posterior_sampling=True,
        random_seed=SEED + 2003 + k * 1009, verbose=False,
    )
    m.fit(X_full, y_full)
    p = m.virtual_ensembles_predict(
        X_test, prediction_type="TotalUncertainty",
        virtual_ensembles_count=10, verbose=False,
    )
    # cols: [Mean, KnowledgeUncertainty, DataUncertainty]
    var_preds.append(np.asarray(p[:, 1] + p[:, 2], dtype=float))
    print(f"  σ-model {k}: var mean={var_preds[-1].mean():.2e}")
var_model = np.mean(var_preds, axis=0)
var_model = np.clip(var_model, 1e-8, None)
print(f"σ² ensemble  mean={var_model.mean():.2e}  median={np.median(var_model):.2e}")

# Kelly sizing: μ / (vol² + σ²_model), then threshold + finalize
test_vol = np.asarray(X_test["vol"].values, dtype=float)
var_vol = np.clip(test_vol ** 2, 1e-8, None)
raw_pos = mu / (var_vol + var_model)

# Threshold on |μ| quantile (matches baseline gating on prediction magnitude)
cutoff = np.quantile(np.abs(mu), THRESHOLD_Q)
raw_pos[np.abs(mu) < cutoff] = 0.0

positions = finalize(raw_pos, shrink_alpha=SHRINK_ALPHA, short_floor=SHORT_FLOOR)
print(f"\nApplied Kelly μ/(vol²+σ²) + threshold(q={THRESHOLD_Q}) "
      f"+ finalize(α={SHRINK_ALPHA}, floor={SHORT_FLOOR})")

submission = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": positions,
})
sub_path = ROOT / "submissions" / "catboost_bars_seed50_kelly.csv"
sub_path.parent.mkdir(exist_ok=True)
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}")
print(submission.describe())
