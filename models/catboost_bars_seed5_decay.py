"""Seed-5 ensemble CatBoost with 4 added time-decayed sentiment features.

Paired-diff evidence: +decay_all Δ=+0.103 (t=+2.02), halves +0.079/+0.127.
Stacked on seed5 (LB 2.583). Hoping for partial transfer.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

from features import (
    SEED, SENT_MAP, DATA_DIR,
    load_train, load_test,
    shape_positions, finalize, SHRINK_ALPHA, SHORT_FLOOR,
)

ROOT = Path(__file__).resolve().parent.parent

BEST_KIND = "thresholded_inv_vol"
N_SPLITS = 5
N_CV_REPEATS = 5
N_SEEDS = 5


def build_decay_sent_features(hdf: pd.DataFrame, all_sessions: np.ndarray) -> pd.DataFrame:
    h = hdf.merge(SENT_MAP, left_on="headline", right_index=True, how="left")
    h["signed"] = h["signed"].fillna(0.0)
    out = pd.DataFrame(index=all_sessions)
    out.index.name = "session"
    for tau in (5, 10, 20):
        w = np.exp(-(49 - h["bar_ix"].to_numpy()) / tau)
        tmp = h.assign(wsig=w * h["signed"].to_numpy())
        out[f"hl_decay_sent_t{tau}"] = tmp.groupby("session")["wsig"].sum()
    recent = h[h["bar_ix"] >= 40]
    out["hl_mean_sent_recent10"] = recent.groupby("session")["signed"].mean()
    return out.fillna(0.0).sort_index()


X_full, y_full = load_train()
X_test = load_test()

headlines_train = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
headlines_test = pd.concat([
    pd.read_parquet(DATA_DIR / "headlines_seen_public_test.parquet"),
    pd.read_parquet(DATA_DIR / "headlines_seen_private_test.parquet"),
], ignore_index=True)

decay_train = build_decay_sent_features(headlines_train, X_full.index.to_numpy())
decay_test = build_decay_sent_features(headlines_test, X_test.index.to_numpy())

X_full = X_full.join(decay_train)
X_test = X_test.join(decay_test)

print(f"Train: {X_full.shape}   Test: {X_test.shape}")
print(f"Added: {list(decay_train.columns)}")

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
print(f"\nEnsemble pred mean={pred_mean.mean():+.5f} std={pred_mean.std():.5f}")

test_vol = np.asarray(X_test["vol"].values, dtype=float)
positions = shape_positions(pred_mean, test_vol, BEST_KIND, threshold_q=0.35)
positions = finalize(positions)
print(f"Applied {BEST_KIND!r} + finalize(α={SHRINK_ALPHA}, floor={SHORT_FLOOR})")

submission = pd.DataFrame({
    "session": X_test.index.astype(int),
    "target_position": positions,
})
sub_path = ROOT / "submissions" / "catboost_bars_seed5_decay.csv"
sub_path.parent.mkdir(exist_ok=True)
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission: {sub_path}")
print(submission.describe())
