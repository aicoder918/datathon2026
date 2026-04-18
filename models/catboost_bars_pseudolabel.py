"""Conservative pseudo-labeling variant of the baseline CatBoost pipeline.

Procedure:
1. Fit an ensemble on the 1k labeled train sessions.
2. Predict all 20k test sessions.
3. Select only the top/bottom decile of test predictions as pseudo-labeled rows.
4. Retrain on train + pseudo rows with reduced weight on pseudo labels.
5. Average a small seed ensemble for the final submission.

This is a distinct class of change from seed averaging and short-suppression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor

from features import (
    SEED, load_train, load_test,
    shape_positions, finalize, SHRINK_ALPHA, SHORT_FLOOR,
)

ROOT = Path(__file__).resolve().parent.parent

BEST_KIND = "thresholded_inv_vol"
FINAL_ITERS = 58
PSEUDO_Q = 0.10
PSEUDO_WEIGHT = 0.20
N_BASE_SEEDS = 10
N_FINAL_SEEDS = 10

X_train, y_train = load_train()
X_test = load_test()
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

print(f"\nStage 1: fitting {N_BASE_SEEDS} base models for pseudo labels ...")
base_preds = []
for k in range(N_BASE_SEEDS):
    m = CatBoostRegressor(
        iterations=FINAL_ITERS,
        learning_rate=0.03,
        depth=5,
        loss_function="MAE",
        random_seed=SEED + k * 1009,
        verbose=False,
    )
    m.fit(X_train, y_train)
    pred = np.asarray(m.predict(X_test), dtype=float)
    base_preds.append(pred)
    print(f"  base seed {k}: pred mean={pred.mean():+.5f} std={pred.std():.5f}")

pred_mean = np.mean(base_preds, axis=0)
q_lo = float(np.quantile(pred_mean, PSEUDO_Q))
q_hi = float(np.quantile(pred_mean, 1.0 - PSEUDO_Q))
mask = (pred_mean <= q_lo) | (pred_mean >= q_hi)

X_pseudo = X_test.loc[mask].copy()
y_pseudo = pd.Series(pred_mean[mask], index=X_pseudo.index, name="y")

print(
    f"\nSelected pseudo rows: {mask.sum()}/{len(mask)} = {mask.mean():.1%} "
    f"(q_lo={q_lo:+.5f}, q_hi={q_hi:+.5f})"
)
print(
    f"Pseudo target stats: mean={y_pseudo.mean():+.5f} std={y_pseudo.std():.5f} "
    f"min={y_pseudo.min():+.5f} max={y_pseudo.max():+.5f}"
)

X_aug = pd.concat([X_train, X_pseudo], axis=0)
y_aug = pd.concat([y_train, y_pseudo], axis=0)
w_aug = np.r_[np.ones(len(X_train), dtype=float),
              np.full(len(X_pseudo), PSEUDO_WEIGHT, dtype=float)]
print(f"Augmented train: {X_aug.shape}   pseudo_weight={PSEUDO_WEIGHT}")

print(f"\nStage 2: fitting {N_FINAL_SEEDS} augmented models ...")
final_preds = []
for k in range(N_FINAL_SEEDS):
    m = CatBoostRegressor(
        iterations=FINAL_ITERS,
        learning_rate=0.03,
        depth=5,
        loss_function="MAE",
        random_seed=SEED + 50000 + k * 1009,
        verbose=False,
    )
    m.fit(X_aug, y_aug, sample_weight=w_aug)
    pred = np.asarray(m.predict(X_test), dtype=float)
    final_preds.append(pred)
    print(f"  final seed {k}: pred mean={pred.mean():+.5f} std={pred.std():.5f}")

final_mean = np.mean(final_preds, axis=0)
print(f"\nAugmented ensemble pred mean={final_mean.mean():+.5f} std={final_mean.std():.5f}")

test_vol = np.asarray(X_test["vol"].values, dtype=float)
positions = shape_positions(final_mean, test_vol, BEST_KIND, threshold_q=0.35)
positions = finalize(positions)
print(f"Applied {BEST_KIND!r} + finalize(α={SHRINK_ALPHA}, floor={SHORT_FLOOR})")

submission = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": positions,
})
sub_path = ROOT / "submissions" / "chatgpt" / "catboost_bars_pseudolabel_q10_w02.csv"
sub_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}")
print(submission.describe())
