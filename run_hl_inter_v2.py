"""HL ridge with expanded interactions: tid×bar, tid×sec, sec×bar, reg×bar."""
from pathlib import Path
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
from run_headline_model import featurize, compute_fwd_returns, RECENCY_TAU, DATA, SUB  # type: ignore
from features import extract_event, SECTORS, REGIONS, N_TEMPLATES  # type: ignore

SECTOR_IDX = {s:i for i,s in enumerate(SECTORS)}
REGION_IDX = {r:i for i,r in enumerate(REGIONS)}
N_SEC = len(SECTORS); N_REG = len(REGIONS)
N_BUCKET = 3

def add_interactions_v2(hdf, bars_seen):
    X_base, bar_ix, sess, _ = featurize(hdf, bars_seen)
    n = X_base.shape[0]
    triples = [extract_event(h) for h in hdf["headline"]]
    tids = np.array([t[0] for t in triples])
    sec_idx = np.array([SECTOR_IDX.get(t[1], -1) for t in triples])
    reg_idx = np.array([REGION_IDX.get(t[2], -1) for t in triples])
    bar_bucket = np.where(bar_ix < 16, 0, np.where(bar_ix < 36, 1, 2))
    tid_v = (tids >= 0); sec_v = sec_idx >= 0; reg_v = reg_idx >= 0
    def onehot(valid, idx, n_classes):
        safe = np.where(valid, idx, 0)
        return csr_matrix((np.where(valid,1.0,0.0),(np.arange(n),safe)), shape=(n, n_classes))
    I_tid_bar = onehot(tid_v, np.where(tid_v, tids*N_BUCKET+bar_bucket, -1)>=0, N_TEMPLATES*N_BUCKET)  # wrong, fix:
    # rebuild properly:
    def cross(a_valid, a_idx, a_n, b_valid, b_idx, b_n):
        v = a_valid & b_valid
        comb = np.where(v, a_idx*b_n + b_idx, 0)
        return csr_matrix((np.where(v,1.0,0.0),(np.arange(n),comb)), shape=(n, a_n*b_n))
    I1 = cross(tid_v, tids, N_TEMPLATES, np.ones(n,bool), bar_bucket, N_BUCKET)
    I2 = cross(tid_v, tids, N_TEMPLATES, sec_v, sec_idx, N_SEC)
    I3 = cross(sec_v, sec_idx, N_SEC, np.ones(n,bool), bar_bucket, N_BUCKET)
    I4 = cross(reg_v, reg_idx, N_REG, np.ones(n,bool), bar_bucket, N_BUCKET)
    X_full = hstack([X_base, I1, I2, I3, I4]).tocsr()
    return X_full, bar_ix, sess

train_seen = pd.read_parquet(DATA/"bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA/"bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA/"bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA/"bars_seen_private_test.parquet")
train_h = pd.read_parquet(DATA/"headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA/"headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA/"headlines_seen_private_test.parquet")

Xtr, _, _ = add_interactions_v2(train_h, train_seen)
Xpub, bar_pub, sess_pub = add_interactions_v2(pub_h, pub_bars)
Xpri, bar_pri, sess_pri = add_interactions_v2(pri_h, pri_bars)
print("v2 feat dim:", Xtr.shape)

fwd = compute_fwd_returns(train_bars, 3)  # K=3 winner
ytr = fwd.reindex(list(zip(train_h["session"].to_numpy(), train_h["bar_ix"].to_numpy()))).fillna(0.0).to_numpy()

for alpha in [3000.0, 10000.0, 30000.0]:
    m = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred_pub = m.predict(Xpub); pred_pri = m.predict(Xpri)
    def agg(pred, bar, sess):
        rec = np.exp(-(49.0-bar)/RECENCY_TAU)
        return pd.DataFrame({"s":sess,"v":pred*rec}).groupby("s")["v"].sum()
    s_pub = agg(pred_pub,bar_pub,sess_pub); s_pri = agg(pred_pri,bar_pri,sess_pri)
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
    nm = f"ridge_hl_interv2_k3_a{int(alpha)}.csv"
    pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/nm,index=False)
    print(f"{nm}: mean={final.mean():.4f}")
