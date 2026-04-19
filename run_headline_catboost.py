"""CatBoost on per-headline features predicting K=5 forward return."""
from pathlib import Path
import sys, numpy as np, pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from run_headline_model import (  # type: ignore
    featurize, compute_fwd_returns, RECENCY_TAU, DATA, SUB,
)

train_seen = pd.read_parquet(DATA/"bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA/"bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA/"bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA/"bars_seen_private_test.parquet")

train_h = pd.read_parquet(DATA/"headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA/"headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA/"headlines_seen_private_test.parquet")

fwd = compute_fwd_returns(train_bars, 5)
Xtr, bar_tr, sess_tr, ytr = featurize(train_h, train_seen, fwd)
Xpub, bar_pub, sess_pub, _ = featurize(pub_h, pub_bars)
Xpri, bar_pri, sess_pri, _ = featurize(pri_h, pri_bars)

Xtr = Xtr.toarray().astype(np.float32)
Xpub = Xpub.toarray().astype(np.float32)
Xpri = Xpri.toarray().astype(np.float32)

# average 5 seeds
preds_pub = np.zeros(len(Xpub)); preds_pri = np.zeros(len(Xpri))
for seed in range(5):
    m = CatBoostRegressor(
        iterations=500, depth=4, learning_rate=0.03,
        l2_leaf_reg=5.0, loss_function="RMSE",
        random_seed=seed, verbose=0,
    ).fit(Xtr, ytr)
    preds_pub += m.predict(Xpub) / 5
    preds_pri += m.predict(Xpri) / 5
    print(f"seed {seed} done")

def agg(pred, bar, sess):
    rec = np.exp(-(49.0 - bar)/RECENCY_TAU)
    return pd.DataFrame({"s":sess,"v":pred*rec}).groupby("s")["v"].sum()
s_pub = agg(preds_pub, bar_pub, sess_pub); s_pri = agg(preds_pri, bar_pri, sess_pri)
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
pos = pred / np.maximum(vol, 1e-6); pos[np.abs(pred)<cutoff]=0.0
mabs=np.mean(np.abs(pos)); scaled=pos/mabs if mabs>0 else pos
final = np.maximum(0.5*scaled + 0.5, 0.30)
pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/"catboost_hl_k5.csv",index=False)
print(f"catboost_hl_k5.csv: mean={final.mean():.4f} std={final.std():.4f} min={final.min():.3f} max={final.max():.3f}")
