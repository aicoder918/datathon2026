"""Single CatBoost train (same CV as catboost_bars.py), then several submissions by
varying threshold_q in shape_positions only. Cheap way to probe the gating lever."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

from features import (
    SEED,
    load_train,
    load_test,
    shape_positions,
    finalize,
    SHRINK_ALPHA,
    SHORT_FLOOR,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_KIND = "thresholded_inv_vol"
N_SPLITS = 5
N_CV_REPEATS = 5
THRESHOLD_GRID = (0.32, 0.33, 0.34, 0.35, 0.36, 0.37)

X_full, y_full = load_train()
X_test = load_test()
print(f"Train: {X_full.shape}   Test: {X_test.shape}")

print(f"\nRunning {N_CV_REPEATS}×{N_SPLITS}-fold CV to pick iterations...")
best_iters: list[int] = []
for repeat in range(N_CV_REPEATS):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat * 97)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full)):
        m = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=5,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=SEED + repeat * 97 + fold,
            early_stopping_rounds=100,
            verbose=False,
        )
        m.fit(
            X_full.iloc[tr_idx],
            y_full.iloc[tr_idx],
            eval_set=(X_full.iloc[va_idx], y_full.iloc[va_idx]),
            use_best_model=True,
        )
        best_iters.append(int(m.tree_count_ or 0))
    print(f"  repeat {repeat}: median best_iters so far = {int(np.median(best_iters))}")

final_iters = max(50, int(np.median(best_iters) * (N_SPLITS / (N_SPLITS - 1))))
print(f"FINAL_ITERS = {final_iters}")

model = CatBoostRegressor(
    iterations=final_iters,
    learning_rate=0.03,
    depth=5,
    loss_function="MAE",
    random_seed=SEED,
    verbose=False,
)
model.fit(X_full, y_full)

pred = np.asarray(model.predict(X_test), dtype=float)
test_vol = np.asarray(X_test["vol"].values, dtype=float)
sessions = X_test.index.to_numpy()

for tq in THRESHOLD_GRID:
    positions = shape_positions(pred, test_vol, BEST_KIND, threshold_q=float(tq))
    positions = finalize(positions)
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": positions,
    })
    name = f"catboost_bars_tq{int(round(tq * 100)):02d}.csv"
    path = OUT_DIR / name
    submission.to_csv(path, index=False)
    print(f"Saved {path}  threshold_q={tq}")

print(f"\nfinalize shrink_alpha={SHRINK_ALPHA} short_floor={SHORT_FLOOR}")
