"""Rank-transform ridge: apply QuantileTransformer (uniform→gaussian) to each
feature column, then Ridge. Different inductive bias from raw-scale ridge:
outliers have bounded leverage and linear relationships in rank-space ≠ linear
in raw-space.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import QuantileTransformer, StandardScaler

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "models"))
from features import load_train, load_test, shape_positions, finalize  # type: ignore

X_tr_df, y_s = load_train()
X_te_df = load_test()
X_tr = X_tr_df.to_numpy(dtype=np.float64)
X_te = X_te_df.to_numpy(dtype=np.float64)
y = y_s.to_numpy(dtype=np.float64)
sessions = X_te_df.index.to_numpy()
vol = X_te_df["vol"].to_numpy(dtype=float)

# fit quantile on TRAIN, transform both
qt = QuantileTransformer(n_quantiles=min(500, len(X_tr)), output_distribution="normal", random_state=42)
X_tr_r = qt.fit_transform(X_tr)
X_te_r = qt.transform(X_te)

# standardize (qt already produces ~N(0,1) but safe)
sc = StandardScaler().fit(X_tr_r)
X_tr_r = sc.transform(X_tr_r)
X_te_r = sc.transform(X_te_r)

for alpha in [300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]:
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha).fit(X_tr_r, y)
    pred = m.predict(X_te_r)
    pos = shape_positions(pred, vol, "thresholded_inv_vol", threshold_q=0.35)
    pos = finalize(pos)
    out = pd.DataFrame({"session": sessions.astype(int), "target_position": pos})
    nm = f"ridge_rank_a{int(alpha)}.csv"
    out.to_csv(ROOT/"submissions"/nm, index=False)
    print(f"{nm:30s} mean={pos.mean():.4f} std={pos.std():.4f} min={pos.min():.3f} max={pos.max():.3f}")

# RidgeCV
rcv = RidgeCV(alphas=np.logspace(-1, 6, 29)).fit(X_tr_r, y)
print("\nRidgeCV alpha =", rcv.alpha_)
pred = rcv.predict(X_te_r)
pos = finalize(shape_positions(pred, vol, "thresholded_inv_vol", threshold_q=0.35))
pd.DataFrame({"session": sessions.astype(int), "target_position": pos}).to_csv(ROOT/"submissions/ridge_rank_cv.csv", index=False)
print(f"ridge_rank_cv.csv mean={pos.mean():.4f} std={pos.std():.4f} min={pos.min():.3f} max={pos.max():.3f}")
