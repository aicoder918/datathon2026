"""Blend raw predictions from multiple ridge points on the same feature path.

This keeps the exact 39-column matrix and the same downstream shaping stack as
the ridge champion, but samples multiple points along the regularization path.

Outputs:
  - submissions/chatgpt/ridge_blend_a1500_strong_w50.csv
  - submissions/chatgpt/ridge_blend_a3000_strong_w50.csv
  - submissions/chatgpt/ridge_blend_a3000_strong_w25.csv
  - submissions/chatgpt/ridge_blend_a3000_strong_w75.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from features import load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_Q = 0.35
RIDGE_ALPHAS_STRONG = np.logspace(1, 6, 16)
BLENDS = [
    # (low_alpha, strong_weight, tag)
    (1500, 0.50, "a1500_strong_w50"),
    (3000, 0.50, "a3000_strong_w50"),
    (3000, 0.25, "a3000_strong_w25"),
    (3000, 0.75, "a3000_strong_w75"),
]


def fit_scaled_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray, float]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    low_preds: dict[int, np.ndarray] = {}
    for alpha in sorted({alpha for alpha, _, _ in BLENDS}):
        model = Ridge(alpha=alpha)
        model.fit(Xtr, y_train)
        low_preds[alpha] = np.asarray(model.predict(Xte), dtype=float)

    strong = RidgeCV(alphas=RIDGE_ALPHAS_STRONG)
    strong.fit(Xtr, y_train)
    strong_pred = np.asarray(strong.predict(Xte), dtype=float)
    return low_preds, strong_pred, float(strong.alpha_)


def save_submission(
    sessions: np.ndarray,
    pred: np.ndarray,
    test_vol: np.ndarray,
    tag: str,
) -> None:
    pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    submission = pd.DataFrame({
        "session": sessions.astype(int),
        "target_position": pos,
    })
    out_path = OUT_DIR / f"ridge_blend_{tag}.csv"
    submission.to_csv(out_path, index=False)
    print(
        f"{tag:<20s} pred mean={pred.mean():+.5f} std={pred.std():.5f} "
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

low_preds, strong_pred, strong_alpha = fit_scaled_predictions(X_train, y_train, X_test)
print(f"strong ridge alpha={strong_alpha:.6f}")

for low_alpha, strong_weight, tag in BLENDS:
    low_weight = 1.0 - strong_weight
    pred = low_weight * low_preds[low_alpha] + strong_weight * strong_pred
    print(
        f"blend {tag}: low_alpha={low_alpha} low_weight={low_weight:.2f} "
        f"strong_weight={strong_weight:.2f}"
    )
    save_submission(sessions, pred, test_vol, tag)
