"""Baseline CatBoost with train-session weights tuned toward the test feature distribution.

Weights are derived from a train-vs-test logistic classifier:
    w(x) = p(test|x) / p(train|x)
then normalized to mean 1 on the train set.

This is a covariate-shift correction attempt. It is a new class of change and
should be evaluated separately from the champion seed ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

from features import (
    SEED, load_train, load_test,
    shape_positions, finalize, SHRINK_ALPHA, SHORT_FLOOR,
)

ROOT = Path(__file__).resolve().parent.parent

BEST_KIND = "thresholded_inv_vol"
N_SPLITS = 5
N_CV_REPEATS = 5
ODDS_CAP = 10.0


def fit_domain_weights(X_train: pd.DataFrame, X_test: pd.DataFrame) -> np.ndarray:
    both = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_domain = np.r_[np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]
    clf = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0),
    )
    clf.fit(both, y_domain)
    p_test = clf.predict_proba(X_train)[:, 1]
    p_test = np.clip(p_test, 1e-4, 1 - 1e-4)
    odds = p_test / (1.0 - p_test)
    odds = np.clip(odds, 1.0 / ODDS_CAP, ODDS_CAP)
    weights = odds / odds.mean()
    return weights


X_full, y_full = load_train()
X_test = load_test()
train_weights = fit_domain_weights(X_full, X_test)
print(f"Train: {X_full.shape}   Test: {X_test.shape}")
print(
    f"Domain weights: mean={train_weights.mean():.4f} std={train_weights.std():.4f} "
    f"min={train_weights.min():.4f} p10={np.quantile(train_weights, 0.1):.4f} "
    f"p90={np.quantile(train_weights, 0.9):.4f} max={train_weights.max():.4f}"
)

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
            sample_weight=train_weights[tr_idx],
            eval_set=(X_full.iloc[va_idx], y_full.iloc[va_idx]),
            use_best_model=True,
        )
        best_iters.append(int(m.tree_count_ or 0))
    print(f"  repeat {repeat}: median best_iters so far = {int(np.median(best_iters))}")

final_iters = max(50, int(np.median(best_iters) * (N_SPLITS / (N_SPLITS - 1))))
print(f"FINAL_ITERS = {final_iters}")

print("\nTraining final weighted model on ALL training data ...")
model = CatBoostRegressor(
    iterations=final_iters, learning_rate=0.03, depth=5,
    loss_function="MAE", random_seed=SEED, verbose=False,
)
model.fit(X_full, y_full, sample_weight=train_weights)

pred = np.asarray(model.predict(X_test), dtype=float)
test_vol = np.asarray(X_test["vol"].values, dtype=float)
positions = shape_positions(pred, test_vol, BEST_KIND, threshold_q=0.35)
positions = finalize(positions)
print(f"Applied {BEST_KIND!r} + finalize(α={SHRINK_ALPHA}, floor={SHORT_FLOOR})")

submission = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": positions,
})
sub_path = ROOT / "submissions" / "catboost_bars_domain_weighted.csv"
sub_path.parent.mkdir(exist_ok=True)
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}")
print(submission.describe())
