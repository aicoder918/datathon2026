"""PLSRegression submission sweep on the full 40-feature matrix.

PLS is a supervised low-rank linear model: it can keep the linear/additive bias
that is working for ridge while compressing correlated inputs into a small
number of predictive components.

Outputs:
  - submissions/chatgpt/pls_c3.csv
  - submissions/chatgpt/pls_c5.csv
  - submissions/chatgpt/pls_c8.csv
  - submissions/chatgpt/pls_c12.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from features import load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_COMPONENTS = [3, 5, 8, 12]
BEST_KIND = "thresholded_inv_vol"
THRESHOLD_Q = 0.35


def fit_predict_pls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
) -> np.ndarray:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(Xtr, y_train)
    pred = np.asarray(model.predict(Xte).reshape(-1), dtype=float)
    return pred


def save_submission(
    sessions: np.ndarray,
    pred: np.ndarray,
    test_vol: np.ndarray,
    n_components: int,
) -> None:
    pos = shape_positions(pred, test_vol, BEST_KIND, threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": pos,
    })
    out_path = OUT_DIR / f"pls_c{n_components}.csv"
    submission.to_csv(out_path, index=False)
    print(
        f"c={n_components:<2d} pred mean={pred.mean():+.5f} std={pred.std():.5f} "
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

for n_components in N_COMPONENTS:
    pred = fit_predict_pls(X_train, y_train, X_test, n_components=n_components)
    save_submission(sessions, pred, test_vol, n_components)
