"""Larger seed ensemble variant but with depth=6, stronger L2, and slower LR.

Uses the same feature set but a different topology to ensure lower correlation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

from features import (
    SEED, load_train, load_test,
    shape_positions, finalize, SHRINK_ALPHA, SHORT_FLOOR,
)

ROOT = Path(__file__).resolve().parent.parent

BEST_KIND = "thresholded_inv_vol"
N_SPLITS = 5
N_CV_REPEATS = 3
N_SEEDS = 20

X_full, y_full = load_train()
X_test = load_test()
print(f"Train: {X_full.shape}   Test: {X_test.shape}")

print(f"\nRunning {N_CV_REPEATS}×{N_SPLITS}-fold CV to pick iterations...")
best_iters: list[int] = []
for repeat in range(N_CV_REPEATS):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat * 97)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full)):
        m = CatBoostRegressor(
            iterations=2000, learning_rate=0.02, depth=6,
            l2_leaf_reg=5.0,
            loss_function="MAE", eval_metric="MAE",
            random_seed=SEED + repeat * 97 + fold,
            early_stopping_rounds=100, verbose=False,
        )
        m.fit(
            X_full.iloc[tr_idx], y_full.iloc[tr_idx],
            eval_set=(X_full.iloc[va_idx], y_full.iloc[va_idx]),
            use_best_model=True,
        )
        best_iters.append(int(m.tree_count_ or 0))
    print(f"  repeat {repeat}: median best_iters so far = {int(np.median(best_iters))}")

final_iters = max(50, int(np.median(best_iters) * (N_SPLITS / (N_SPLITS - 1))))
print(f"FINAL_ITERS = {final_iters}")

print(f"\nFitting {N_SEEDS} seed-diversified CatBoosts on ALL training data ...")
preds = []
for k in range(N_SEEDS):
    m = CatBoostRegressor(
        iterations=final_iters, learning_rate=0.02, depth=6,
        l2_leaf_reg=5.0,
        loss_function="MAE", random_seed=SEED + k * 1009, verbose=False,
    )
    m.fit(X_full, y_full)
    pred = np.asarray(m.predict(X_test), dtype=float)
    preds.append(pred)
    print(f"  seed {k:02d}: pred mean={pred.mean():+.5f} std={pred.std():.5f}")

pred_mean = np.mean(preds, axis=0)
print(
    f"\nEnsemble pred mean={pred_mean.mean():+.5f} std={pred_mean.std():.5f} "
    f"(per-seed std range: {min(p.std() for p in preds):.5f}–{max(p.std() for p in preds):.5f})"
)

test_vol = np.asarray(X_test["vol"].values, dtype=float)
positions = shape_positions(pred_mean, test_vol, BEST_KIND, threshold_q=0.35)
positions = finalize(positions)
print(f"Applied {BEST_KIND!r} + finalize(α={SHRINK_ALPHA}, floor={SHORT_FLOOR})")

submission = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": positions,
})
sub_path = ROOT / "submissions" / "catboost_bars_depth6.csv"
sub_path.parent.mkdir(exist_ok=True)
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}")
print(submission.describe())
