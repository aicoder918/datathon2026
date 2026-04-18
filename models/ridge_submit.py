"""Low-capacity linear submissions on the production feature set.

Generates two candidate submissions:
  - ridge_all: RidgeCV on all production features
  - ridge_top10: RidgeCV on top-10 |corr(x, y)| features computed on train only

This is a genuinely different inductive bias from CatBoost: no interactions,
no tree structure, just a regularized linear map on the same features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from features import (
    load_train,
    load_test,
    shape_positions,
    finalize,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_ALPHAS = np.logspace(-3, 3, 13)
BEST_KIND = "thresholded_inv_vol"
THRESHOLD_Q = 0.35


def select_topk(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    corrs = np.array([
        abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 1e-12 else 0.0
        for j in range(X.shape[1])
    ])
    return np.argsort(-corrs)[:k]


def fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(Xtr, y_train)
    pred = np.asarray(model.predict(Xte), dtype=float)
    return pred, float(model.alpha_)


def save_submission(
    pred: np.ndarray,
    test_vol: np.ndarray,
    sessions: np.ndarray,
    out_name: str,
) -> pd.Series:
    pos = shape_positions(pred, test_vol, BEST_KIND, threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": pos,
    })
    out_path = OUT_DIR / out_name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission.describe())
    return pd.Series(pos, index=sessions)


X_train_df, y_train_s = load_train()
X_test_df = load_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

X_train = X_train_df.to_numpy(dtype=np.float64)
X_test = X_test_df.to_numpy(dtype=np.float64)
y_train = y_train_s.to_numpy(dtype=np.float64)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy()

pred_all, alpha_all = fit_predict_ridge(X_train, y_train, X_test)
print(f"ridge_all alpha={alpha_all:.6f} pred mean={pred_all.mean():+.5f} std={pred_all.std():.5f}")
save_submission(pred_all, test_vol, sessions, "ridge_all.csv")

idx10 = select_topk(X_train, y_train, 10)
top10_cols = list(X_train_df.columns[idx10])
pred_top10, alpha_top10 = fit_predict_ridge(X_train[:, idx10], y_train, X_test[:, idx10])
print(
    f"\nridge_top10 alpha={alpha_top10:.6f} pred mean={pred_top10.mean():+.5f} "
    f"std={pred_top10.std():.5f}"
)
print(f"top10 cols: {top10_cols}")
save_submission(pred_top10, test_vol, sessions, "ridge_top10.csv")
