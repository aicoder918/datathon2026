"""Low-capacity linear submissions on the production feature set.

Generates candidate submissions:
  - ridge_all: RidgeCV on all production features
  - ridge_top10: RidgeCV on top-10 |corr(x, y)| features computed on train only
  - ridge_top5: same as top10 but top-5 features only (lower capacity / different blend geometry)
  - enet_top10: ElasticNetCV on the same top-10 features as ridge_top10 (L1+L2 path)

This is a genuinely different inductive bias from CatBoost: no interactions,
no tree structure, just a regularized linear map on the same features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from features import (
    SEED,
    load_train,
    load_test,
    shape_positions,
    finalize,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_ALPHAS = np.logspace(-3, 3, 13)
ENET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
# Explicit grid silences sklearn FutureWarning for ElasticNetCV (alphas=None deprecated).
ENET_ALPHAS = np.logspace(-5, 0.5, 36)
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


def fit_predict_enet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = ElasticNetCV(
        l1_ratio=ENET_L1_RATIOS,
        alphas=ENET_ALPHAS,
        cv=3,
        max_iter=5000,
        n_jobs=1,
        random_state=SEED,
    )
    model.fit(Xtr, y_train)
    pred = np.asarray(model.predict(Xte), dtype=float)
    return pred, float(model.alpha_), float(model.l1_ratio_)


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

idx5 = select_topk(X_train, y_train, 5)
top5_cols = list(X_train_df.columns[idx5])
pred_top5, alpha_top5 = fit_predict_ridge(X_train[:, idx5], y_train, X_test[:, idx5])
print(
    f"\nridge_top5 alpha={alpha_top5:.6f} pred mean={pred_top5.mean():+.5f} "
    f"std={pred_top5.std():.5f}"
)
print(f"top5 cols: {top5_cols}")
save_submission(pred_top5, test_vol, sessions, "ridge_top5.csv")

pred_enet10, alpha_enet10, l1_enet10 = fit_predict_enet(
    X_train[:, idx10], y_train, X_test[:, idx10]
)
print(
    f"\nenet_top10 alpha={alpha_enet10:.6f} l1_ratio={l1_enet10:.6f} "
    f"pred mean={pred_enet10.mean():+.5f} std={pred_enet10.std():.5f}"
)
save_submission(pred_enet10, test_vol, sessions, "enet_top10.csv")
