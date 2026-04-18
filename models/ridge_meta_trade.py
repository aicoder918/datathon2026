"""Two-stage tradability model.

Stage 1 predicts whether a session is worth trading (large |y|).
Stage 2 regresses y only on those tradable sessions.
Final prediction = p(tradable) * conditional_return_estimate.

Output:
  - submissions/chatgpt/ridge_meta_trade_q65.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from features import load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADE_Q = 0.65
RIDGE_ALPHA = 3000.0
THRESHOLD_Q = 0.35


X_train_df, y_train_s = load_train()
X_test_df = load_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

y_train = y_train_s.to_numpy(dtype=np.float64)
trade_cut = float(np.quantile(np.abs(y_train), TRADE_Q))
trade_label = (np.abs(y_train) >= trade_cut).astype(int)
print(f"trade cutoff |y| >= {trade_cut:.6f}; tradable fraction={trade_label.mean():.3f}")

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_df.to_numpy(dtype=np.float64))
Xte = scaler.transform(X_test_df.to_numpy(dtype=np.float64))

gate = LogisticRegression(
    max_iter=4000,
    class_weight="balanced",
    C=1.0,
    random_state=42,
)
gate.fit(Xtr, trade_label)
p_trade = gate.predict_proba(Xte)[:, 1]

reg = Ridge(alpha=RIDGE_ALPHA)
reg.fit(Xtr[trade_label == 1], y_train[trade_label == 1])
pred_trade = np.asarray(reg.predict(Xte), dtype=float)
pred = p_trade * pred_trade

test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
pos = finalize(pos)

submission = pd.DataFrame({
    "session": X_test_df.index.astype(int),
    "target_position": pos,
})
out_path = OUT_DIR / f"ridge_meta_trade_q{int(TRADE_Q * 100):02d}.csv"
submission.to_csv(out_path, index=False)
print(
    f"p_trade mean={p_trade.mean():.4f} pred_trade std={pred_trade.std():.5f} "
    f"pred std={pred.std():.5f}"
)
print(f"\nSaved submission: {out_path}")
print(submission["target_position"].describe().to_string())
