"""Meta-learner: session-level features derived from per-headline k3 predictions
+ bar features, trained on session-level target.
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

train_seen = pd.read_parquet(DATA/"bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA/"bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA/"bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA/"bars_seen_private_test.parquet")
train_h = pd.read_parquet(DATA/"headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA/"headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA/"headlines_seen_private_test.parquet")

fwd = compute_fwd_returns(train_bars, 3)
ytr_h = fwd.reindex(list(zip(train_h["session"].to_numpy(), train_h["bar_ix"].to_numpy()))).fillna(0.0).to_numpy()
Xtr, bar_tr, sess_tr = add_interactions(train_h, train_seen)
Xpub, bar_pub, sess_pub = add_interactions(pub_h, pub_bars)
Xpri, bar_pri, sess_pri = add_interactions(pri_h, pri_bars)

# Get OOF train predictions for session-level stacking
print("computing OOF train predictions ...")
oof_preds = np.zeros(len(ytr_h))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in kf.split(np.arange(len(ytr_h))):
    Xtr_f = Xtr[tr_idx]; ytr_f = ytr_h[tr_idx]
    # triple-ens
    oof_sub = np.zeros(len(va_idx))
    for alpha in [3000.0, 10000.0, 30000.0]:
        m = Ridge(alpha=alpha).fit(Xtr_f, ytr_f)
        oof_sub += m.predict(Xtr[va_idx]) / 3
    oof_preds[va_idx] = oof_sub

# Fit final triple-ens on all train for test predictions
full_models = [Ridge(alpha=a).fit(Xtr, ytr_h) for a in [3000.0, 10000.0, 30000.0]]
test_pred_pub = np.mean([m.predict(Xpub) for m in full_models], axis=0)
test_pred_pri = np.mean([m.predict(Xpri) for m in full_models], axis=0)

# Session-level stacking features
def build_stack_features(per_h_pred, bars, sessions):
    df = pd.DataFrame({"session":sessions, "bar_ix":bars, "pred":per_h_pred})
    df["rec"] = np.exp(-(49.0-df["bar_ix"])/RECENCY_TAU)
    df["pred_rec"] = df["pred"]*df["rec"]
    g = df.groupby("session")
    out = pd.DataFrame({
        "sum_pred_rec": g["pred_rec"].sum(),
        "mean_pred_rec": g["pred_rec"].mean(),
        "max_pred_rec": g["pred_rec"].max(),
        "min_pred_rec": g["pred_rec"].min(),
        "std_pred": g["pred"].std().fillna(0.0),
        "count_hl": g.size(),
        "sum_pos": g["pred_rec"].apply(lambda x: x[x>0].sum()),
        "sum_neg": g["pred_rec"].apply(lambda x: x[x<0].sum()),
        "last_bar": g["bar_ix"].max(),
        "mean_bar": g["bar_ix"].mean(),
    })
    return out

train_sessions = train_seen["session"].unique()
pub_sessions = pub_bars["session"].unique()
pri_sessions = pri_bars["session"].unique()

Ftr = build_stack_features(oof_preds, train_h["bar_ix"].to_numpy(), train_h["session"].to_numpy()).reindex(train_sessions).fillna(0.0)
Fpub = build_stack_features(test_pred_pub, bar_pub, sess_pub).reindex(pub_sessions).fillna(0.0)
Fpri = build_stack_features(test_pred_pri, bar_pri, sess_pri).reindex(pri_sessions).fillna(0.0)

# Session-level target (the one we actually want)
seen_train = train_seen
unseen_train = pd.read_parquet(DATA/"bars_unseen_train.parquet")
last_seen = seen_train.sort_values("bar_ix").groupby("session")["close"].last()
last_unseen = unseen_train.sort_values("bar_ix").groupby("session")["close"].last()
y_sess = (last_unseen/last_seen - 1).reindex(train_sessions)

# Combine with existing features via models.features if desired. For now use stack only.
sc = StandardScaler().fit(Ftr.values)
Xtr_s = sc.transform(Ftr.values); Xpub_s = sc.transform(Fpub.values); Xpri_s = sc.transform(Fpri.values)

rcv = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr_s, y_sess.values)
print("stack RidgeCV alpha:", rcv.alpha_)
pred_pub = rcv.predict(Xpub_s); pred_pri = rcv.predict(Xpri_s)

def sess_vol(bars):
    b=bars.sort_values(["session","bar_ix"]).copy()
    b["bar_ret"]=b.groupby("session")["close"].pct_change().fillna(0.0)
    return b.groupby("session")["bar_ret"].std()

ref = pd.read_csv(ROOT/"submissions/chatgpt/ridge_top10.csv").sort_values("session").reset_index(drop=True)
sess_all = ref["session"].values
full_pred = pd.Series(0.0, index=sess_all)
full_pred.loc[pub_sessions] = pred_pub
full_pred.loc[pri_sessions] = pred_pri
vol = pd.concat([sess_vol(pub_bars), sess_vol(pri_bars)]).reindex(sess_all).to_numpy()
pred = full_pred.to_numpy()
cutoff = np.quantile(np.abs(pred), 0.35)
pos = pred/np.maximum(vol,1e-6); pos[np.abs(pred)<cutoff]=0.0
mabs=np.mean(np.abs(pos)); scaled=pos/mabs if mabs>0 else pos
final = np.maximum(0.5*scaled+0.5, 0.30)
pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/"stack_hl_session_ridge.csv",index=False)
print(f"stack_hl_session_ridge: mean={final.mean():.4f}")
