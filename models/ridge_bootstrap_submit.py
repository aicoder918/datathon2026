"""Bootstrap-averaged ridge on the 40 base features.

Ridge is deterministic for a given train set, so seed averaging does nothing.
But bootstrap resampling — fitting ridge on B different [sampled-with-replacement]
views of the 1000 train rows — produces B genuinely different coefficient
vectors. Averaging test predictions over them is the ridge analog of CatBoost
seed ensembling, which was THE one lever that transferred (+0.013 LB).

Same 40 features, same α as ridge_all_strong (2.667 champion). Only change is
replacing one Ridge fit with the mean of B=200 bootstrap fits. Coefficient
vector stays in the robust 40-feature subspace — no new dims, no rotation into
train-noise directions.

Outputs per alpha so we can LB-compare α choice separately from bootstrap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from features import load_train, load_test, shape_positions, finalize

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [10000, 30000]
B = 200
SEED = 42
THRESHOLD_Q = 0.35


X_train_df, y_train_s = load_train()
X_test_df = load_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

X_train = X_train_df.to_numpy(dtype=np.float64)
X_test = X_test_df.to_numpy(dtype=np.float64)
y_train = y_train_s.to_numpy(dtype=np.float64)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy().astype(int)
n_train = X_train.shape[0]

scaler = StandardScaler()
Xtr_std = scaler.fit_transform(X_train)
Xte_std = scaler.transform(X_test)

rng = np.random.default_rng(SEED)
bootstrap_idx = [rng.integers(0, n_train, n_train) for _ in range(B)]

for alpha in ALPHAS:
    preds = np.zeros(len(X_test), dtype=np.float64)
    for b, idx in enumerate(bootstrap_idx):
        m = Ridge(alpha=alpha)
        m.fit(Xtr_std[idx], y_train[idx])
        preds += m.predict(Xte_std)
        if (b + 1) % 50 == 0:
            print(f"  alpha={alpha}  fit {b+1}/{B}")
    pred = preds / B
    pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos = finalize(pos)
    sub = pd.DataFrame({"session": sessions, "target_position": pos})
    out = OUT_DIR / f"ridge_bootstrap_B{B}_a{int(alpha)}.csv"
    sub.to_csv(out, index=False)
    print(f"alpha={alpha:<6}  pred mean={pred.mean():+.5f} std={pred.std():.5f}")
    print(f"  {out.name}  pos mean={pos.mean():.4f} std={pos.std():.4f} min={pos.min():.4f} max={pos.max():.4f}")
