"""Train the CatBoost bars+headlines model on all training data and write the
submission. Hyperparameters (CatBoost + post-processing) are set from the
winning trial of hyperparam_search.py. Holdout/Sharpe diagnostics live in
evaluate.py."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor

from features import (
    SEED, load_train, load_test,
    shape_positions, finalize,
)

ROOT = Path(__file__).resolve().parent.parent

# --- tuned by hyperparam_search.py (30 random trials over 5 splits) ---
# Best mean raw Sharpe = 3.102 vs constant-long reference 3.098 — the model
# contributes a small long-biased tilt; most of the score is the long drift.
BEST_KIND = "sign"
BEST_SHRINK_ALPHA = 0.05   # 95% constant-long + 5% model tilt
BEST_SHORT_FLOOR = 0.0
BEST_DEPTH = 5
BEST_LR = 0.03
BEST_L2 = 5.0
BEST_ITERS = 1200

# ---------- build features ----------
X_full, y_full = load_train()
X_test = load_test()
print(f"Train: {X_full.shape}   Test: {X_test.shape}")

# ---------- fit on ALL training data with search-tuned hyperparams ----------
print(f"\nTraining final submission model on ALL {len(X_full)} sessions "
      f"(depth={BEST_DEPTH}, lr={BEST_LR}, l2={BEST_L2}, iters={BEST_ITERS}) ...")
model = CatBoostRegressor(
    iterations=BEST_ITERS, learning_rate=BEST_LR, depth=BEST_DEPTH,
    l2_leaf_reg=BEST_L2,
    loss_function="RMSE", random_seed=SEED, verbose=False,
)
model.fit(X_full, y_full)

# ---------- predict + shape positions ----------
pred = np.asarray(model.predict(X_test), dtype=float)
test_vol = np.asarray(X_test["vol"].values, dtype=float)
positions = shape_positions(pred, test_vol, BEST_KIND)
positions = finalize(positions, shrink_alpha=BEST_SHRINK_ALPHA,
                    short_floor=BEST_SHORT_FLOOR)
print(f"Applied {BEST_KIND!r} + finalize(shrink α={BEST_SHRINK_ALPHA}, "
      f"floor={BEST_SHORT_FLOOR})")

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
