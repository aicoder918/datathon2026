"""CatBoost trained ONLY on the top 10 Ridge features to generate a heavily decorrelated signal.
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

# Features extracted as having highest correlation with target linearly
TOP10_COLS = [
    'event_impact_recent_k3', 'event_impact_sec_recent_k3', 'max_drawdown',
    'event_impact_recent_k10', 'event_impact_sec_recent_k10', 'event_impact_sec_recent_k5',
    'range_mean', 'min_low', 'event_impact_recent_k5', 'vol'
]

X_full_df, y_full = load_train()
X_test_df = load_test()

X_full = X_full_df[TOP10_COLS].copy()
X_test = X_test_df[TOP10_COLS].copy()

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
        iterations=final_iters, learning_rate=0.03, depth=5,
        loss_function="MAE", random_seed=SEED + k * 1009, verbose=False,
    )
    m.fit(X_full, y_full)
    pred = np.asarray(m.predict(X_test), dtype=float)
    preds.append(pred)

pred_mean = np.mean(preds, axis=0)

test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
positions = shape_positions(pred_mean, test_vol, BEST_KIND, threshold_q=0.35)
positions = finalize(positions)

submission = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": positions,
})
sub_path = ROOT / "submissions" / "chatgpt" / "catboost_top10.csv"
sub_path.parent.mkdir(exist_ok=True, parents=True)
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}")
print(submission.describe())
