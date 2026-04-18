"""Focused untried ridge pred-space follow-up blends.

Outputs:
  - submissions/chatgpt/ridge_blend_a3000_strong_w33.csv
  - submissions/chatgpt/ridge_blend_a5000_strong_w50.csv
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
VARIANTS = [
    (3000.0, 1.0 / 3.0, "ridge_blend_a3000_strong_w33.csv"),
    (5000.0, 0.50, "ridge_blend_a5000_strong_w50.csv"),
]


X_train_df, y_train_s = load_train()
X_test_df = load_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_df.to_numpy(dtype=np.float64))
Xte = scaler.transform(X_test_df.to_numpy(dtype=np.float64))
y_train = y_train_s.to_numpy(dtype=np.float64)

strong = RidgeCV(alphas=RIDGE_ALPHAS_STRONG)
strong.fit(Xtr, y_train)
pred_strong = np.asarray(strong.predict(Xte), dtype=float)
print(f"strong alpha={float(strong.alpha_):.6f}")

cache: dict[float, np.ndarray] = {}
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy()

for alpha, strong_weight, out_name in VARIANTS:
    if alpha not in cache:
        m = Ridge(alpha=alpha)
        m.fit(Xtr, y_train)
        cache[alpha] = np.asarray(m.predict(Xte), dtype=float)
    pred = (1.0 - strong_weight) * cache[alpha] + strong_weight * pred_strong
    pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    submission = pd.DataFrame({"session": sessions.astype(int), "target_position": pos})
    out_path = OUT_DIR / out_name
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission: {out_path}")
    print(submission["target_position"].describe().to_string())
