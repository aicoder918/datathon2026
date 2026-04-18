"""Seed-ensemble variant of catboost_bars: train 5 CatBoosts with different
random_seeds on ALL data, average their predictions, then shape as production.

Rationale: big4 paired-diff showed Δ=+0.128 (t=+3.75, both halves positive).
Seed ensembling reduces prediction variance without changing what the model
learns — the one GBM lever that tends to transfer LB-wise.

Uses the same CV-picked iterations as production.
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
N_CV_REPEATS = 5
N_SEEDS = 5

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

print(f"\nFitting {N_SEEDS} seed-diversified CatBoosts on ALL training data ...")
preds = []
for k in range(N_SEEDS):
    m = CatBoostRegressor(
        iterations=final_iters, learning_rate=0.03, depth=5,
        loss_function="MAE", random_seed=SEED + k * 1009, verbose=False,
    )
    m.fit(X_full, y_full)
    preds.append(np.asarray(m.predict(X_test), dtype=float))
    print(f"  seed {k}: pred mean={preds[-1].mean():+.5f} std={preds[-1].std():.5f}")

pred_mean = np.mean(preds, axis=0)
print(f"\nEnsemble pred mean={pred_mean.mean():+.5f} std={pred_mean.std():.5f} "
      f"(per-seed std range: "
      f"{min(p.std() for p in preds):.5f}–{max(p.std() for p in preds):.5f})")

test_vol = np.asarray(X_test["vol"].values, dtype=float)
positions = shape_positions(pred_mean, test_vol, BEST_KIND, threshold_q=0.35)
positions = finalize(positions)
print(f"Applied {BEST_KIND!r} + finalize(α={SHRINK_ALPHA}, floor={SHORT_FLOOR})")

submission = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": positions,
})
sub_path = ROOT / "submissions" / "catboost_bars_seed5.csv"
sub_path.parent.mkdir(exist_ok=True)
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}")
print(submission.describe())
