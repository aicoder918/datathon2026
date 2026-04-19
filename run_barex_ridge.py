"""Ridge on expanded bar-only feature set (no headlines).
Features per session (50 bars):
  - 50 per-bar log returns
  - Rolling vol in windows (5,10,20)
  - Slope of close over first/second half
  - Running drawdown/peak ratios
Target: forward return (last_unseen / last_seen - 1)
Goal: produce a low-correlation orthogonal branch vs the current champion.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent

def build(bars_seen: pd.DataFrame) -> pd.DataFrame:
    df = bars_seen.sort_values(["session","bar_ix"]).copy()
    df["bar_ret"] = df.groupby("session")["close"].pct_change().fillna(0.0)
    # pivot: sessions x 50 returns
    pv = df.pivot(index="session", columns="bar_ix", values="bar_ret").sort_index()
    pv.columns = [f"r{c:02d}" for c in pv.columns]
    # windowed vol
    for w in (5,10,20):
        pv[f"vol_tail{w}"] = pv.iloc[:, -w:].std(axis=1)
        pv[f"vol_head{w}"] = pv.iloc[:, :w].std(axis=1)
        pv[f"ret_tail{w}"] = pv.iloc[:, -w:].sum(axis=1)
        pv[f"ret_head{w}"] = pv.iloc[:, :w].sum(axis=1)
    # slope halves
    x = np.arange(50, dtype=float)
    ret_cols = [c for c in pv.columns if c.startswith("r")]
    mat = pv[ret_cols].values
    cm = np.cumsum(mat, axis=1)
    # slope of cumsum over first 25 and last 25
    slope1 = (cm[:,24] - cm[:,0]) / 24.0
    slope2 = (cm[:,49] - cm[:,25]) / 24.0
    pv["slope_first25"] = slope1
    pv["slope_last25"] = slope2
    pv["total_ret"] = cm[:,49]
    pv["max_dd"] = np.min(mat.cumsum(axis=1) - np.maximum.accumulate(mat.cumsum(axis=1), axis=1), axis=1)
    # sign streaks
    signs = np.sign(mat)
    pos_streak = np.max(np.maximum.reduce([signs==1, np.zeros_like(signs,dtype=bool)]).astype(int).cumsum(axis=1), axis=1)
    pv["n_pos_bars"] = (signs>0).sum(axis=1)
    pv["n_neg_bars"] = (signs<0).sum(axis=1)
    return pv

seen = pd.read_parquet(ROOT/"data/bars_seen_train.parquet")
unseen = pd.read_parquet(ROOT/"data/bars_unseen_train.parquet")
last_seen = seen.sort_values("bar_ix").groupby("session")["close"].last()
last_unseen = unseen.sort_values("bar_ix").groupby("session")["close"].last()
y = (last_unseen/last_seen - 1).sort_index()

X_tr = build(seen).reindex(y.index)
test_seen = pd.concat([
    pd.read_parquet(ROOT/"data/bars_seen_public_test.parquet"),
    pd.read_parquet(ROOT/"data/bars_seen_private_test.parquet"),
], ignore_index=True)
X_te = build(test_seen)

ref = pd.read_csv(ROOT/"submissions/chatgpt/ridge_top10.csv").sort_values("session").reset_index(drop=True)
ref_sess = ref["session"].values
X_te = X_te.reindex(ref_sess).values
X_tr_v = X_tr.values; y_v = y.values

# volatility on test from seen bars (use existing feature in the matrix? we have vol_tail20 etc.)
# For shaping we need a "vol" per session; take vol_tail10 from build output
vol_te = pd.DataFrame(X_te, columns=X_tr.columns)["vol_tail10"].values

sc = StandardScaler().fit(X_tr_v)
X_tr_s = sc.transform(X_tr_v)
X_te_s = sc.transform(X_te)

from sklearn.linear_model import Ridge
def shape_and_save(pred, nm):
    cutoff = np.quantile(np.abs(pred), 0.35)
    pos = pred / np.maximum(vol_te, 1e-6)
    pos[np.abs(pred)<cutoff] = 0.0
    m=np.mean(np.abs(pos)); scaled=pos/m if m>0 else pos
    final = np.maximum(0.5*scaled + 0.5, 0.30)
    pd.DataFrame({"session":ref_sess,"target_position":final}).to_csv(ROOT/"submissions"/nm,index=False)
    print(f"{nm:30s} mean={final.mean():.4f} std={final.std():.4f} min={final.min():.3f} max={final.max():.3f}")
    return final

for alpha in [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]:
    m = Ridge(alpha=alpha).fit(X_tr_s, y_v)
    pred = m.predict(X_te_s)
    shape_and_save(pred, f"ridge_barex_a{int(alpha)}.csv")

rcv = RidgeCV(alphas=np.logspace(-1, 6, 29)).fit(X_tr_s, y_v)
print("RidgeCV alpha =", rcv.alpha_)
shape_and_save(rcv.predict(X_te_s), "ridge_barex_cv.csv")
