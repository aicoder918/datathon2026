"""Semi-supervised HL ridge: train on train + pseudo-labeled public headlines.
Pseudo-labels from the triple-ensemble predictions.
"""
from pathlib import Path
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.sparse import vstack

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

Xtr, _, _ = add_interactions(train_h, train_seen)
Xpub, bar_pub, sess_pub = add_interactions(pub_h, pub_bars)
Xpri, bar_pri, sess_pri = add_interactions(pri_h, pri_bars)

# Use the triple-ens model predictions as pseudo-labels for pub headlines
# First refit models to get per-headline preds (not aggregated):
models = [Ridge(alpha=a).fit(Xtr, ytr) for a in [3000.0, 10000.0, 30000.0]]
preds_pub_by_a = [m.predict(Xpub) for m in models]
y_pub_pseudo = np.mean(preds_pub_by_a, axis=0)

# Train augmented model on train + pub with weight 0.3 on pub
W = np.concatenate([np.ones(len(ytr)), 0.3 * np.ones(len(y_pub_pseudo))])
X_aug = vstack([Xtr, Xpub]).tocsr()
y_aug = np.concatenate([ytr, y_pub_pseudo])

for alpha in [10000.0, 30000.0]:
    m = Ridge(alpha=alpha).fit(X_aug, y_aug, sample_weight=W)
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
    nm = f"ridge_hl_inter_ssl_a{int(alpha)}.csv"
    pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/nm,index=False)
    print(f"{nm}: mean={final.mean():.4f}")
