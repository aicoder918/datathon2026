"""HL ridge with interaction features: tid × bar_ix bucket, sector × sentiment,
template × $-scale, etc. Goal: capture non-linear response that basic one-hots miss.
"""
from pathlib import Path
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from run_headline_model import (  # type: ignore
    featurize, compute_fwd_returns, RECENCY_TAU, DATA, SUB,
)
sys.path.insert(0, str(ROOT / "models"))
from features import extract_event, SENT_MAP, N_TEMPLATES, SECTORS, REGIONS  # type: ignore

def add_interactions(hdf, bars_seen):
    X_base, bar_ix, sess, _ = featurize(hdf, bars_seen)
    n = X_base.shape[0]
    # rebuild tid/sec arrays
    triples = [extract_event(h) for h in hdf["headline"]]
    tids = np.array([t[0] for t in triples])
    secs = [t[1] for t in triples]
    SECTOR_IDX = {s:i for i,s in enumerate(SECTORS)}
    sec_idx = np.array([SECTOR_IDX.get(s, -1) for s in secs])
    # tid × bar_bucket (buckets: early 0-15, mid 16-35, late 36-49)
    bar_bucket = np.where(bar_ix < 16, 0, np.where(bar_ix < 36, 1, 2))
    tid_valid = (tids >= 0)
    sec_valid = sec_idx >= 0
    # tid x bar_bucket (shape: n x (N_TEMPLATES * 3))
    cross_tid_bar = np.where(tid_valid, tids * 3 + bar_bucket, -1)
    mask1 = cross_tid_bar >= 0
    I1 = csr_matrix(
        (np.where(mask1,1.0,0.0), (np.arange(n), np.where(mask1, cross_tid_bar, 0))),
        shape=(n, N_TEMPLATES*3)
    )
    # tid x sector (shape: n x (N_TEMPLATES * N_SEC))
    N_SEC = len(SECTORS)
    cross_tid_sec = np.where(tid_valid & sec_valid, tids*N_SEC + sec_idx, -1)
    mask2 = cross_tid_sec >= 0
    I2 = csr_matrix(
        (np.where(mask2,1.0,0.0), (np.arange(n), np.where(mask2, cross_tid_sec, 0))),
        shape=(n, N_TEMPLATES*N_SEC)
    )
    X_full = hstack([X_base, I1, I2]).tocsr()
    return X_full, bar_ix, sess

train_seen = pd.read_parquet(DATA/"bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA/"bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA/"bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA/"bars_seen_private_test.parquet")

train_h = pd.read_parquet(DATA/"headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA/"headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA/"headlines_seen_private_test.parquet")

fwd = compute_fwd_returns(train_bars, 5)
Xtr, bar_tr, sess_tr = add_interactions(train_h, train_seen)
ytr = fwd.reindex(list(zip(train_h["session"].to_numpy(), train_h["bar_ix"].to_numpy()))).fillna(0.0).to_numpy()
Xpub, bar_pub, sess_pub = add_interactions(pub_h, pub_bars)
Xpri, bar_pri, sess_pri = add_interactions(pri_h, pri_bars)
print("train:",Xtr.shape,"pub:",Xpub.shape,"pri:",Xpri.shape)

# standardize last 8 dense cols of the BASE features (cols -8 of base = at offset total-8-?
# Actually with interactions added after, trailing cols are one-hots. Don't standardize.
# Leave as-is.

for alpha in [3000.0, 10000.0, 30000.0]:
    m = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred_pub = m.predict(Xpub); pred_pri = m.predict(Xpri)
    def agg(pred, bar, sess):
        rec = np.exp(-(49.0-bar)/RECENCY_TAU)
        return pd.DataFrame({"s":sess,"v":pred*rec}).groupby("s")["v"].sum()
    s_pub = agg(pred_pub, bar_pub, sess_pub); s_pri = agg(pred_pri, bar_pri, sess_pri)
    ref = pd.read_csv(ROOT/"submissions/chatgpt/ridge_top10.csv").sort_values("session").reset_index(drop=True)
    sess_all = ref["session"].values
    score = pd.Series(0.0, index=sess_all)
    score.loc[s_pub.index] = s_pub.values; score.loc[s_pri.index] = s_pri.values
    def sess_vol(bars):
        b=bars.sort_values(["session","bar_ix"]).copy()
        b["bar_ret"]=b.groupby("session")["close"].pct_change().fillna(0.0)
        return b.groupby("session")["bar_ret"].std()
    vol = pd.concat([sess_vol(pub_bars), sess_vol(pri_bars)]).reindex(sess_all).to_numpy()
    pred = score.to_numpy()
    cutoff = np.quantile(np.abs(pred), 0.35)
    pos = pred/np.maximum(vol,1e-6); pos[np.abs(pred)<cutoff]=0.0
    mabs=np.mean(np.abs(pos)); scaled=pos/mabs if mabs>0 else pos
    final = np.maximum(0.5*scaled+0.5, 0.30)
    nm = f"ridge_hl_inter_a{int(alpha)}.csv"
    pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/nm,index=False)
    print(f"{nm}: mean={final.mean():.4f} std={final.std():.4f} min={final.min():.3f} max={final.max():.3f}")
