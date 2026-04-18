"""Kill-shorts variant of catboost_bars: zero out all negative raw predictions
before shaping. Everything else identical to production.

Rationale: asym_shape sweep found the only positive direction was tq_neg→1
(essentially long-only). Drift is positive and unconditional; short-side model
confidence is noisy. Testing on LB whether removing all shorts helps.
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

print("\nTraining final model on ALL training data ...")
model = CatBoostRegressor(
    iterations=final_iters, learning_rate=0.03, depth=5,
    loss_function="MAE", random_seed=SEED, verbose=False,
)
model.fit(X_full, y_full)

pred = np.asarray(model.predict(X_test), dtype=float)

# --- KILL SHORTS: zero out negative predictions before shaping ---
n_neg_before = int((pred < 0).sum())
pred_long_only = pred.copy()
pred_long_only[pred_long_only < 0] = 0.0
print(f"Killed {n_neg_before}/{len(pred)} negative preds "
      f"({n_neg_before/len(pred):.1%}) — remaining set is long-only + zeros")

test_vol = np.asarray(X_test["vol"].values, dtype=float)
positions = shape_positions(pred_long_only, test_vol, BEST_KIND, threshold_q=0.35)
positions = finalize(positions)
print(f"Applied {BEST_KIND!r} + finalize(α={SHRINK_ALPHA}, floor={SHORT_FLOOR})")

submission = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": positions,
})
sub_path = ROOT / "submissions" / "catboost_bars_killshorts.csv"
sub_path.parent.mkdir(exist_ok=True)
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}")
print(submission.describe())
