"""Fixed-alpha ridge submission sweep around the current strong-ridge winner.

Outputs one submission per alpha into submissions/chatgpt/:
  ridge_alpha_1500.csv
  ridge_alpha_3000.csv
  ridge_alpha_5000.csv
  ridge_alpha_10000.csv
  ridge_alpha_20000.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features import load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [1500, 3000, 5000, 10000, 20000]
BEST_KIND = "thresholded_inv_vol"
THRESHOLD_Q = 0.35


def fit_predict_fixed_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = Ridge(alpha=alpha)
    model.fit(Xtr, y_train)
    return np.asarray(model.predict(Xte), dtype=float)


def save_submission(
    sessions: np.ndarray,
    pred: np.ndarray,
    test_vol: np.ndarray,
    alpha: float,
) -> None:
    pos = shape_positions(pred, test_vol, BEST_KIND, threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": pos,
    })
    out_path = OUT_DIR / f"ridge_alpha_{int(alpha)}.csv"
    submission.to_csv(out_path, index=False)
    print(
        f"alpha={alpha:<5.0f} pred mean={pred.mean():+.5f} std={pred.std():.5f} "
        f"-> {out_path.name}"
    )
    print(submission["target_position"].describe().to_string())


X_train_df, y_train_s = load_train()
X_test_df = load_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

X_train = X_train_df.to_numpy(dtype=np.float64)
X_test = X_test_df.to_numpy(dtype=np.float64)
y_train = y_train_s.to_numpy(dtype=np.float64)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy()

for alpha in ALPHAS:
    pred = fit_predict_fixed_ridge(X_train, y_train, X_test, alpha=alpha)
    save_submission(sessions, pred, test_vol, alpha)
