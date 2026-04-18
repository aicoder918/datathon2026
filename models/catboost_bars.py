"""Train the CatBoost bars+headlines model on all training data and write the
submission. Uses a thorough repeated K-fold CV pass to pick the right number
of boosting iterations before the final fit. Holdout/Sharpe diagnostics live
in `evaluate.py`.

(Optuna-tuned variant tried 2026-04-18 scored 2.51-2.55 on LB vs 2.57 baseline —
confirmed overfit to train-session distribution; reverted.)"""
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
N_CV_REPEATS = 5   # 25 early-stopped fits total; uses median best_iters

# ---------- build features ----------
X_full, y_full = load_train()
X_test = load_test()
print(f"Train: {X_full.shape}   Test: {X_test.shape}")

# ---------- CV to pick FINAL_ITERS ----------
print(f"\nRunning {N_CV_REPEATS}×{N_SPLITS}-fold CV with early stopping to pick iterations...")
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

# Final model trains on 100% of data vs N_SPLITS-1/N_SPLITS in CV, so scale up.
final_iters = max(50, int(np.median(best_iters) * (N_SPLITS / (N_SPLITS - 1))))
print(f"\nbest_iters distribution: min={min(best_iters)} "
      f"median={int(np.median(best_iters))} max={max(best_iters)}")
print(f"FINAL_ITERS (median * 5/4) = {final_iters}")

# ---------- fit on ALL training data ----------
print("\nTraining final submission model on ALL training data ...")
model = CatBoostRegressor(
    iterations=final_iters, learning_rate=0.03, depth=5,
    loss_function="MAE", random_seed=SEED, verbose=False,
)
model.fit(X_full, y_full)

# ---------- predict + shape positions ----------
pred = np.asarray(model.predict(X_test), dtype=float)
test_vol = np.asarray(X_test["vol"].values, dtype=float)
positions = shape_positions(pred, test_vol, BEST_KIND, threshold_q=0.35)
positions = finalize(positions)
print(f"Applied {BEST_KIND!r} + finalize(shrink α={SHRINK_ALPHA}, floor={SHORT_FLOOR})")

submission = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": positions,
})
sub_path = ROOT / "submissions" / "catboost_bars.csv"
sub_path.parent.mkdir(exist_ok=True)
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}")
print(submission.head())
print(submission.describe())
