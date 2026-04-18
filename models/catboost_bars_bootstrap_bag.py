"""Bootstrap-bagged CatBoost submission.

Distinct from the seed ensemble: each model trains on a bootstrap resample of
the 1k training sessions, then predictions are averaged. This targets train-set
idiosyncrasy rather than only model-seed noise.

Outputs:
  - submissions/chatgpt/catboost_bars_bootstrap30.csv
  - submissions/chatgpt/catboost_bars_bootstrap30_killshorts_blend.csv
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
N_BAGS = 30


def fit_bootstrap_predictions(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    n_bags: int = N_BAGS,
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n = len(X_full)
    preds: list[np.ndarray] = []
    for bag in range(n_bags):
        sample_idx = rng.integers(0, n, n)
        uniq = int(np.unique(sample_idx).size)
        model = CatBoostRegressor(
            iterations=FINAL_ITERS,
            learning_rate=0.03,
            depth=5,
            loss_function="MAE",
            random_seed=SEED + bag * 1009,
            verbose=False,
        )
        model.fit(X_full.iloc[sample_idx], y_full.iloc[sample_idx])
        pred = np.asarray(model.predict(X_test), dtype=float)
        preds.append(pred)
        print(
            f"  bag {bag:02d}: unique_train={uniq}/{n} "
            f"pred mean={pred.mean():+.5f} std={pred.std():.5f}"
        )
    return np.mean(preds, axis=0)


X_full, y_full = load_train()
X_test = load_test()
print(f"Train: {X_full.shape}   Test: {X_test.shape}")
print(f"Bootstrap bags: {N_BAGS}   FINAL_ITERS={FINAL_ITERS}")

print(f"\nFitting {N_BAGS} bootstrap CatBoosts on resampled training sessions ...")
pred_mean = fit_bootstrap_predictions(X_full, y_full, X_test, n_bags=N_BAGS)
print(f"\nBootstrap ensemble pred mean={pred_mean.mean():+.5f} std={pred_mean.std():.5f}")

test_vol = np.asarray(X_test["vol"].values, dtype=float)
base_pos = shape_positions(pred_mean, test_vol, BEST_KIND, threshold_q=0.35)
base_pos = finalize(base_pos)

pred_kill = pred_mean.copy()
pred_kill[pred_kill < 0] = 0.0
kill_pos = shape_positions(pred_kill, test_vol, BEST_KIND, threshold_q=0.35)
kill_pos = finalize(kill_pos)
blend_pos = 0.5 * base_pos + 0.5 * kill_pos

print(
    f"Applied {BEST_KIND!r} + finalize(α={SHRINK_ALPHA}, floor={SHORT_FLOOR}); "
    f"killed {(pred_mean < 0).sum()}/{len(pred_mean)} negative averaged preds"
)

out_dir = ROOT / "submissions" / "chatgpt"
out_dir.mkdir(parents=True, exist_ok=True)

base_sub = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": base_pos,
})
base_path = out_dir / "catboost_bars_bootstrap30.csv"
base_sub.to_csv(base_path, index=False)
print(f"\nSaved submission: {base_path}")
print(base_sub.describe())

blend_sub = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": blend_pos,
})
blend_path = out_dir / "catboost_bars_bootstrap30_killshorts_blend.csv"
blend_sub.to_csv(blend_path, index=False)
print(f"\nSaved submission: {blend_path}")
print(blend_sub.describe())
