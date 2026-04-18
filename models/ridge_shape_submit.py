"""Lower short_floor on top of strong-α ridge — let the ridge tails breathe.

Motivation: bootstrap ridge compressed min 0.30→0.50 (shorts muted) and
max 2.89→2.35 (extreme longs muted) and lost −0.045 LB. That says ridge's
tail positions are real signal, not noise. Direction of travel: LESS
compression, not more. Lower short_floor gives legitimate shorts more room.

We don't know exactly which α the 2.667 champion `ridge_all_strong.csv`
used, so produce α=10000 at three floors — one control (0.30) that should
roughly match champion stats to confirm the α, and two experiments (0.20,
0.10). Also α=20000 with floor=0.20 as a combined push further along the
monotone α trend + tails-breathe lever.
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

VARIANTS = [
    # (alpha, shrink_alpha, short_floor, tag)
    (10000, 0.50, 0.30, "a10000_sa050"),   # control — matches prior pipeline
    (10000, 0.70, 0.30, "a10000_sa070"),   # more ridge tilt
    (10000, 0.60, 0.30, "a10000_sa060"),
    (10000, 0.40, 0.30, "a10000_sa040"),   # less ridge tilt
    (10000, 0.30, 0.30, "a10000_sa030"),
]
THRESHOLD_Q = 0.35


X_train_df, y_train_s = load_train()
X_test_df = load_test()
print(f"Train: {X_train_df.shape}   Test: {X_test_df.shape}")

X_train = X_train_df.to_numpy(dtype=np.float64)
X_test = X_test_df.to_numpy(dtype=np.float64)
y_train = y_train_s.to_numpy(dtype=np.float64)
test_vol = np.asarray(X_test_df["vol"].values, dtype=float)
sessions = X_test_df.index.to_numpy().astype(int)

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

fits: dict[int, np.ndarray] = {}

for alpha, sa, floor, tag in VARIANTS:
    if alpha not in fits:
        m = Ridge(alpha=alpha)
        m.fit(Xtr, y_train)
        fits[alpha] = np.asarray(m.predict(Xte), dtype=float)
    pred = fits[alpha]
    pos = shape_positions(pred, test_vol, "thresholded_inv_vol", threshold_q=THRESHOLD_Q)
    pos = finalize(pos, shrink_alpha=sa, short_floor=floor)
    sub = pd.DataFrame({"session": sessions, "target_position": pos})
    out = OUT_DIR / f"ridge_shape_{tag}.csv"
    sub.to_csv(out, index=False)
    print(f"α={alpha:<5} sa={sa:.2f} floor={floor:.2f}  pred mean={pred.mean():+.5f} std={pred.std():.5f}")
    print(f"  {out.name}  pos mean={pos.mean():.4f} std={pos.std():.4f} min={pos.min():.4f} max={pos.max():.4f}")
