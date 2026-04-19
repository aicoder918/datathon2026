"""HL classifier: logistic regression on sign(fwd K=5 return).
Predicts probability of positive direction, converted to signed signal (2*p-1).
Different loss from ridge → potentially different signal.
"""
from pathlib import Path
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from run_headline_interactions import add_interactions  # type: ignore
from run_headline_model import compute_fwd_returns, RECENCY_TAU, DATA, SUB  # type: ignore

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
y_sign = (ytr > 0).astype(int)

Xtr, bar_tr, sess_tr = add_interactions(train_h, train_seen)
Xpub, bar_pub, sess_pub = add_interactions(pub_h, pub_bars)
Xpri, bar_pri, sess_pri = add_interactions(pri_h, pri_bars)

print("fit logreg ...")
m = LogisticRegression(C=1e-4, max_iter=500, solver="liblinear").fit(Xtr, y_sign)
pp = m.predict_proba(Xpub)[:,1]
pr = m.predict_proba(Xpri)[:,1]
# convert to signed signal
signed_pub = 2*pp - 1
signed_pri = 2*pr - 1

def agg(pred, bar, sess):
    rec = np.exp(-(49.0-bar)/RECENCY_TAU)
    return pd.DataFrame({"s":sess,"v":pred*rec}).groupby("s")["v"].sum()
s_pub = agg(signed_pub, bar_pub, sess_pub); s_pri = agg(signed_pri, bar_pri, sess_pri)

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
pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/"logreg_hl_k5_inter.csv",index=False)
print(f"logreg_hl_k5_inter: mean={final.mean():.4f} std={final.std():.4f} min={final.min():.3f} max={final.max():.3f}")
