"""Local analog retrieval model using half-session path shape plus tabular context.

For each test session, retrieve nearest historical analogs and predict from
their realized forward returns.

Output:
  - submissions/chatgpt/analog_knn_k25.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from features import DATA_DIR, load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "chatgpt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 25
THRESHOLD_Q = 0.35


def build_path_matrix(train_sessions: np.ndarray, test_sessions: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_seen = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    test_seen = pd.concat([
        pd.read_parquet(DATA_DIR / "bars_seen_public_test.parquet"),
        pd.read_parquet(DATA_DIR / "bars_seen_private_test.parquet"),
    ], ignore_index=True)

    def pivot_path(df: pd.DataFrame, sessions: np.ndarray) -> pd.DataFrame:
        piv = df.pivot(index="session", columns="bar_ix", values="close").reindex(sessions)
        base = piv[0].to_numpy(dtype=float).reshape(-1, 1)
        norm = piv.to_numpy(dtype=float) / np.maximum(base, 1e-12) - 1.0
        cols = [f"path_close_{i}" for i in piv.columns]
        return pd.DataFrame(norm, index=sessions, columns=cols)

    return pivot_path(train_seen, train_sessions), pivot_path(test_seen, test_sessions)


X_train_df, y_train_s = load_train()
X_test_df = load_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

train_path, test_path = build_path_matrix(X_train_df.index.to_numpy(), X_test_df.index.to_numpy())
extra_cols = [
    "hl_n", "hl_n_recent", "hl_net_sent", "hl_net_sent_recent",
    "event_impact_k5", "event_impact_recent_k5", "event_impact_sec_k5",
]
X_train_knn = train_path.join(X_train_df[extra_cols])
X_test_knn = test_path.join(X_test_df[extra_cols])

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train_knn.to_numpy(dtype=np.float64))
Xte = scaler.transform(X_test_knn.to_numpy(dtype=np.float64))
y_train = y_train_s.to_numpy(dtype=np.float64)

nn = NearestNeighbors(n_neighbors=K, metric="euclidean")
nn.fit(Xtr)
dist, idx = nn.kneighbors(Xte, return_distance=True)
weights = 1.0 / np.maximum(dist, 1e-6)
pred = (weights * y_train[idx]).sum(axis=1) / weights.sum(axis=1)

test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
pos = finalize(pos)

submission = pd.DataFrame({
    "session": X_test_df.index.astype(int),
    "target_position": pos,
})
out_path = OUT_DIR / f"analog_knn_k{K}.csv"
submission.to_csv(out_path, index=False)
print(f"pred mean={pred.mean():+.5f} std={pred.std():.5f}")
print(f"\nSaved submission: {out_path}")
print(submission["target_position"].describe().to_string())
