"""HL ridge with added session-context: running count of prior headlines in
session, running avg sentiment, running net sentiment, + the interaction features.
"""
from pathlib import Path
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from run_headline_interactions import add_interactions  # type: ignore
from run_headline_model import compute_fwd_returns, RECENCY_TAU, DATA, SUB  # type: ignore
sys.path.insert(0, str(ROOT / "models"))
from features import SENT_MAP  # type: ignore

def add_ctx(hdf: pd.DataFrame):
    """Return (n_prior_in_session, cumsum_sentiment_before)."""
    h = hdf.copy()
    h["_sent"] = h["headline"].map(SENT_MAP["signed"].to_dict()).fillna(0.0)
    h = h.sort_values(["session","bar_ix"]).reset_index(drop=False)
    # running cumsums within session
    h["cum_n_prior"] = h.groupby("session").cumcount()
    h["cum_sent_prior"] = h.groupby("session")["_sent"].cumsum() - h["_sent"]
    h["mean_sent_prior"] = np.where(h["cum_n_prior"]>0, h["cum_sent_prior"]/np.maximum(h["cum_n_prior"],1), 0.0)
    h = h.sort_values("index").reset_index(drop=True)  # restore original order
    return h[["cum_n_prior","cum_sent_prior","mean_sent_prior"]].to_numpy(float)

def build(hdf, bars_seen):
    X_inter, bar_ix, sess = add_interactions(hdf, bars_seen)
    ctx = add_ctx(hdf)
    X = hstack([X_inter, csr_matrix(ctx)]).tocsr()
    return X, bar_ix, sess

train_seen = pd.read_parquet(DATA/"bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA/"bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA/"bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA/"bars_seen_private_test.parquet")

train_h = pd.read_parquet(DATA/"headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA/"headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA/"headlines_seen_private_test.parquet")

fwd = compute_fwd_returns(train_bars, 5)
ytr = fwd.reindex(list(zip(train_h["session"].to_numpy(), train_h["bar_ix"].to_numpy()))).fillna(0.0).to_numpy()

Xtr, bar_tr, sess_tr = build(train_h, train_seen)
Xpub, bar_pub, sess_pub = build(pub_h, pub_bars)
Xpri, bar_pri, sess_pri = build(pri_h, pri_bars)
print("train:",Xtr.shape)

# standardize trailing 3 ctx cols only
def zs(X, mu, sd):
    X = X.toarray().astype(np.float64, copy=True)
    X[:,-3:] = (X[:,-3:] - mu)/np.where(sd<1e-8,1.0,sd)
    return X
mu = np.asarray(Xtr[:,-3:].mean(axis=0)).ravel()
sd = np.asarray(np.sqrt(((Xtr[:,-3:].toarray()-mu)**2).mean(axis=0))).ravel()
Xtr_d = zs(Xtr,mu,sd); Xpub_d = zs(Xpub,mu,sd); Xpri_d = zs(Xpri,mu,sd)

for alpha in [10000.0, 30000.0]:
    m = Ridge(alpha=alpha).fit(Xtr_d, ytr)
    pred_pub = m.predict(Xpub_d); pred_pri = m.predict(Xpri_d)
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
    nm = f"ridge_hl_interctx_a{int(alpha)}.csv"
    pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/nm,index=False)
    print(f"{nm}: mean={final.mean():.4f} std={final.std():.4f} min={final.min():.3f} max={final.max():.3f}")
