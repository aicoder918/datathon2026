"""Robust ridge submissions on the production 40-feature matrix.

Idea:
- winsorize each feature using train quantiles
- scale with RobustScaler instead of StandardScaler
- fit strongly regularized ridge

Outputs:
  - submissions/chatgpt/ridge_robust_q01.csv   (clip to 1/99 pct)
  - submissions/chatgpt/ridge_robust_q02.csv   (clip to 2/98 pct)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler

from features import load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = np.logspace(1, 6, 16)
BEST_KIND = "thresholded_inv_vol"
THRESHOLD_Q = 0.35


def winsorize_from_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    lower_q: float,
    upper_q: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lo = X_train.quantile(lower_q)
    hi = X_train.quantile(upper_q)
    Xtr = X_train.clip(lower=lo, upper=hi, axis=1)
    Xte = X_test.clip(lower=lo, upper=hi, axis=1)
    return Xtr, Xte


def fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    Xtr = scaler.fit_transform(X_train.to_numpy(dtype=np.float64))
    Xte = scaler.transform(X_test.to_numpy(dtype=np.float64))
    model = RidgeCV(alphas=ALPHAS)
    model.fit(Xtr, y_train.to_numpy(dtype=np.float64))
    pred = np.asarray(model.predict(Xte), dtype=float)
    return pred, float(model.alpha_)


def save_submission(
    sessions: np.ndarray,
    pred: np.ndarray,
    test_vol: np.ndarray,
    name: str,
) -> None:
    pos = shape_positions(pred, test_vol, BEST_KIND, threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": pos,
    })
    out_path = OUT_DIR / name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission.describe())


X_train, y_train = load_train()
X_test = load_test()
sessions = X_test.index.to_numpy()
test_vol = np.asarray(X_test["vol"].values, dtype=float)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

for tag, (lq, uq) in {
    "q01": (0.01, 0.99),
    "q02": (0.02, 0.98),
}.items():
    Xtr_w, Xte_w = winsorize_from_train(X_train, X_test, lq, uq)
    pred, alpha = fit_predict(Xtr_w, y_train, Xte_w)
    print(
        f"{tag} clip=({lq:.2f},{uq:.2f}) alpha={alpha:.6f} "
        f"pred mean={pred.mean():+.5f} std={pred.std():.5f}"
    )
    save_submission(sessions, pred, test_vol, f"ridge_robust_{tag}.csv")
