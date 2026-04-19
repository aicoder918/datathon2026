"""HL ridge with interactions + company one-hot."""
from pathlib import Path
import sys, re, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
from run_headline_interactions import add_interactions
from run_headline_model import compute_fwd_returns, RECENCY_TAU, DATA, SUB

_COMPANY_RE = re.compile(r"^([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,2})")
def extract_company(h: str) -> str:
    m = _COMPANY_RE.match(h)
    return m.group(1) if m else ""

train_seen = pd.read_parquet(DATA/"bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA/"bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA/"bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA/"bars_seen_private_test.parquet")
train_h = pd.read_parquet(DATA/"headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA/"headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA/"headlines_seen_private_test.parquet")

# Build company vocab from train (train-only to avoid overfit, but apply to all)
train_comp = [extract_company(h) for h in train_h["headline"]]
pub_comp = [extract_company(h) for h in pub_h["headline"]]
pri_comp = [extract_company(h) for h in pri_h["headline"]]
all_companies = sorted(set(train_comp) | set(pub_comp) | set(pri_comp))
comp_idx = {c:i for i,c in enumerate(all_companies) if c}
print(f"n companies: {len(all_companies)}")

def comp_onehot(comps, n_rows):
    valid = np.array([c in comp_idx for c in comps])
    safe = np.array([comp_idx.get(c, 0) for c in comps])
    return csr_matrix(
        (np.where(valid,1.0,0.0), (np.arange(n_rows), safe)),
        shape=(n_rows, len(all_companies)),
    )

Xtr_inter, bar_tr, sess_tr = add_interactions(train_h, train_seen)
Xpub_inter, bar_pub, sess_pub = add_interactions(pub_h, pub_bars)
Xpri_inter, bar_pri, sess_pri = add_interactions(pri_h, pri_bars)

Xtr = hstack([Xtr_inter, comp_onehot(train_comp, Xtr_inter.shape[0])]).tocsr()
Xpub = hstack([Xpub_inter, comp_onehot(pub_comp, Xpub_inter.shape[0])]).tocsr()
Xpri = hstack([Xpri_inter, comp_onehot(pri_comp, Xpri_inter.shape[0])]).tocsr()
print("train dim:", Xtr.shape)

fwd = compute_fwd_returns(train_bars, 3)
ytr = fwd.reindex(list(zip(train_h["session"].to_numpy(), train_h["bar_ix"].to_numpy()))).fillna(0.0).to_numpy()

for alpha in [3000.0, 10000.0, 30000.0, 100000.0]:
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
    nm = f"ridge_hl_interco_k3_a{int(alpha)}.csv"
    pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/nm,index=False)
    print(f"{nm}: mean={final.mean():.4f}")
