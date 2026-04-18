"""Soft-gated regime mixture-of-experts on the production feature matrix.

Cluster sessions into coarse regimes, fit one ridge expert per regime, then use
a gating classifier to produce soft test-time expert weights.

Output:
  - submissions/chatgpt/ridge_regime_moe_k4.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from features import load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REGIMES = 4
ALPHA = 3000.0
THRESHOLD_Q = 0.35
REGIME_COLS = [
    "seen_ret", "mom_3", "mom_10", "vol", "vol_recent",
    "max_drawdown", "ret_skew", "hl_n", "hl_n_recent", "hl_net_sent",
]


X_train_df, y_train_s = load_train()
X_test_df = load_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

regime_scaler = StandardScaler()
Ztr = regime_scaler.fit_transform(X_train_df[REGIME_COLS].to_numpy(dtype=np.float64))
Zte = regime_scaler.transform(X_test_df[REGIME_COLS].to_numpy(dtype=np.float64))

kmeans = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=42)
train_regimes = kmeans.fit_predict(Ztr)
gate = LogisticRegression(
    max_iter=4000,
    C=1.0,
    random_state=42,
)
gate.fit(Ztr, train_regimes)
gate_probs = gate.predict_proba(Zte)

full_scaler = StandardScaler()
Xtr = full_scaler.fit_transform(X_train_df.to_numpy(dtype=np.float64))
Xte = full_scaler.transform(X_test_df.to_numpy(dtype=np.float64))
y_train = y_train_s.to_numpy(dtype=np.float64)

expert_preds = np.zeros((N_REGIMES, len(X_test_df)), dtype=np.float64)
for r in range(N_REGIMES):
    mask = train_regimes == r
    n = int(mask.sum())
    print(f"regime {r}: n_train={n}")
    model = Ridge(alpha=ALPHA)
    model.fit(Xtr[mask], y_train[mask])
    expert_preds[r] = model.predict(Xte)

pred = np.sum(gate_probs.T * expert_preds, axis=0)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
pos = finalize(pos)

submission = pd.DataFrame({
    "session": X_test_df.index.astype(int),
    "target_position": pos,
})
out_path = OUT_DIR / f"ridge_regime_moe_k{N_REGIMES}.csv"
submission.to_csv(out_path, index=False)
print(f"pred mean={pred.mean():+.5f} std={pred.std():.5f}")
print(f"\nSaved submission: {out_path}")
print(submission["target_position"].describe().to_string())
