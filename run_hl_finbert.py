"""HL ridge with FinBERT CLS embeddings (per-headline, 768d) + interactions."""
from pathlib import Path
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
from run_headline_interactions import add_interactions
from run_headline_model import compute_fwd_returns, RECENCY_TAU, DATA, SUB

EMB = pd.read_parquet(ROOT/"artifacts/finbert_cls_embeddings.parquet").set_index("headline")
emb_cols = [c for c in EMB.columns if c.startswith("emb_")]
# PCA down for dimensionality
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=32, random_state=42)
EMB_mat = EMB[emb_cols].values
print("fitting SVD on embeddings ...")
svd.fit(EMB_mat)
EMB_svd = svd.transform(EMB_mat)
EMB_df = pd.DataFrame(EMB_svd, index=EMB.index, columns=[f"ec_{i}" for i in range(32)])

def join_emb(hdf):
    e = EMB_df.reindex(hdf["headline"].values).fillna(0.0).values
    return e

train_seen = pd.read_parquet(DATA/"bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA/"bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA/"bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA/"bars_seen_private_test.parquet")
train_h = pd.read_parquet(DATA/"headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA/"headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA/"headlines_seen_private_test.parquet")

Xtr_i, bar_tr, sess_tr = add_interactions(train_h, train_seen)
Xpub_i, bar_pub, sess_pub = add_interactions(pub_h, pub_bars)
Xpri_i, bar_pri, sess_pri = add_interactions(pri_h, pri_bars)

Etr = join_emb(train_h); Epub = join_emb(pub_h); Epri = join_emb(pri_h)
# standardize embeddings based on train
emu = Etr.mean(axis=0); esd = Etr.std(axis=0) + 1e-8
Etr = (Etr - emu) / esd; Epub = (Epub - emu) / esd; Epri = (Epri - emu) / esd

Xtr = hstack([Xtr_i, csr_matrix(Etr)]).tocsr()
Xpub = hstack([Xpub_i, csr_matrix(Epub)]).tocsr()
Xpri = hstack([Xpri_i, csr_matrix(Epri)]).tocsr()
print("train dim:", Xtr.shape)

fwd = compute_fwd_returns(train_bars, 3)
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
    nm = f"ridge_hl_interfb_k3_a{int(alpha)}.csv"
    pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/nm,index=False)
    print(f"{nm}: mean={final.mean():.4f}")
