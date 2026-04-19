"""Rich meta-learner: ridge on 40 session features + session-aggregated HL predictions.
Uses OOF HL predictions on train to avoid leakage.
"""
from pathlib import Path
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
from run_headline_interactions import add_interactions  # type: ignore
from run_headline_model import compute_fwd_returns, RECENCY_TAU, DATA, SUB  # type: ignore
from features import load_train, load_test, shape_positions, finalize  # type: ignore

train_seen = pd.read_parquet(DATA/"bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA/"bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA/"bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA/"bars_seen_private_test.parquet")
train_h = pd.read_parquet(DATA/"headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA/"headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA/"headlines_seen_private_test.parquet")

fwd_k3 = compute_fwd_returns(train_bars, 3)
ytr_h = fwd_k3.reindex(list(zip(train_h["session"].to_numpy(), train_h["bar_ix"].to_numpy()))).fillna(0.0).to_numpy()

Xtr_h, bar_tr, sess_tr = add_interactions(train_h, train_seen)
Xpub_h, bar_pub, sess_pub = add_interactions(pub_h, pub_bars)
Xpri_h, bar_pri, sess_pri = add_interactions(pri_h, pri_bars)

# OOF per-headline predictions for train (triple ens)
print("OOF HL predictions ...")
oof_preds = np.zeros(len(ytr_h))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in kf.split(np.arange(len(ytr_h))):
    for alpha in [3000.0, 10000.0, 30000.0]:
        m = Ridge(alpha=alpha).fit(Xtr_h[tr_idx], ytr_h[tr_idx])
        oof_preds[va_idx] += m.predict(Xtr_h[va_idx]) / 3
# Full-train models for test
full_models = [Ridge(alpha=a).fit(Xtr_h, ytr_h) for a in [3000.0,10000.0,30000.0]]
test_pred_pub = np.mean([m.predict(Xpub_h) for m in full_models], axis=0)
test_pred_pri = np.mean([m.predict(Xpri_h) for m in full_models], axis=0)

def stack_feats(preds, bars, sess):
    df = pd.DataFrame({"session":sess,"bar_ix":bars,"pred":preds})
    df["rec"] = np.exp(-(49.0-df["bar_ix"])/RECENCY_TAU)
    df["pr"] = df["pred"]*df["rec"]
    g = df.groupby("session")
    out = pd.DataFrame({
        "hl_sum_pr": g["pr"].sum(),
        "hl_mean_pr": g["pr"].mean(),
        "hl_max_pr": g["pr"].max(),
        "hl_min_pr": g["pr"].min(),
        "hl_std_pred": g["pred"].std().fillna(0.0),
        "hl_sum_pos_pr": g["pr"].apply(lambda x: x[x>0].sum()),
        "hl_sum_neg_pr": g["pr"].apply(lambda x: x[x<0].sum()),
        "hl_count": g.size(),
        "hl_last_bar_pred": df.sort_values("bar_ix").groupby("session")["pred"].last(),
    })
    return out

# 40-feature set
X_tr40, y_s = load_train()
X_te40 = load_test()

# train sessions order from load_train
train_sessions = X_tr40.index.to_numpy()
pub_sessions = X_te40.loc[X_te40.index<11000].index.to_numpy()
pri_sessions = X_te40.loc[X_te40.index>=11000].index.to_numpy()

Ftr_stack = stack_feats(oof_preds, train_h["bar_ix"].to_numpy(), train_h["session"].to_numpy()).reindex(train_sessions).fillna(0.0)
Fpub_stack = stack_feats(test_pred_pub, bar_pub, sess_pub).reindex(pub_sessions).fillna(0.0)
Fpri_stack = stack_feats(test_pred_pri, bar_pri, sess_pri).reindex(pri_sessions).fillna(0.0)

# Combine
X_tr_full = pd.concat([X_tr40, Ftr_stack], axis=1).values
X_te_full = pd.concat([X_te40, pd.concat([Fpub_stack, Fpri_stack])], axis=1).reindex(X_te40.index).values

sc = StandardScaler().fit(X_tr_full)
Xtr_s = sc.transform(X_tr_full); Xte_s = sc.transform(X_te_full)

rcv = RidgeCV(alphas=np.logspace(-1,6,22)).fit(Xtr_s, y_s.values)
print("rich meta alpha:", rcv.alpha_)
pred = rcv.predict(Xte_s)

sessions = X_te40.index.to_numpy()
vol = X_te40["vol"].to_numpy()
cutoff = np.quantile(np.abs(pred), 0.35)
pos = pred/np.maximum(vol,1e-6); pos[np.abs(pred)<cutoff]=0.0
mabs=np.mean(np.abs(pos)); scaled=pos/mabs if mabs>0 else pos
final = np.maximum(0.5*scaled+0.5, 0.30)
pd.DataFrame({"session":sessions.astype(int),"target_position":final}).to_csv(SUB/"ridge_rich_meta.csv",index=False)
print(f"ridge_rich_meta: mean={final.mean():.4f}")

# also dump at shrink 0.3/0.7 for variety
for sa in [0.3, 0.7]:
    f = np.maximum(sa*scaled + (1-sa)*1.0, 0.30)
    pd.DataFrame({"session":sessions.astype(int),"target_position":f}).to_csv(SUB/f"ridge_rich_meta_sa{int(sa*10):02d}.csv",index=False)
    print(f"sa={sa}: mean={f.mean():.4f}")
