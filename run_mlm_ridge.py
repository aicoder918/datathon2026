"""Ridge on MLM embeddings only, save submission for blending.
Train target = last_unseen/last_seen - 1 on train sessions (tr:0..tr:999).
Predict on pub:* and priv:* sessions → 20000 rows.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
emb = pd.read_parquet(ROOT/"artifacts/bar_mlm_embeddings.parquet")
feat_cols = [c for c in emb.columns if c.startswith("bemb_")]
emb = emb.sort_values("sid").reset_index(drop=True)
is_tr = emb["sid"].str.startswith("tr:")
is_pub = emb["sid"].str.startswith("pub:")
is_pri = emb["sid"].str.startswith("priv:")

X_tr = emb.loc[is_tr, feat_cols].to_numpy()
X_pub = emb.loc[is_pub, feat_cols].to_numpy()
X_pri = emb.loc[is_pri, feat_cols].to_numpy()
pub_sess = emb.loc[is_pub, "sid"].str.slice(4).astype(int).to_numpy()
pri_sess = emb.loc[is_pri, "sid"].str.slice(5).astype(int).to_numpy()

# build train target
seen = pd.read_parquet(ROOT/"data/bars_seen_train.parquet")
unseen = pd.read_parquet(ROOT/"data/bars_unseen_train.parquet")
last_seen = seen.sort_values("bar_ix").groupby("session")["close"].last()
last_unseen = unseen.sort_values("bar_ix").groupby("session")["close"].last()
y = (last_unseen/last_seen - 1).sort_index().values
# train X in tr: order: tr:0..tr:999 — should already align since sid sorted alphabetically puts tr:0 first
# but string sort: tr:10 before tr:2. Let's key properly:
tr_sids = emb.loc[is_tr, "sid"].tolist()
tr_idx = np.argsort([int(s[3:]) for s in tr_sids])
X_tr = X_tr[tr_idx]
# now y is indexed by numeric train session id 0..999, aligned with X_tr

# Also build vol from test bars (for shape)
pub_bars = pd.read_parquet(ROOT/"data/bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(ROOT/"data/bars_seen_private_test.parquet")
def sess_vol(df):
    df=df.sort_values(["session","bar_ix"]).copy()
    df["bar_ret"]=df.groupby("session")["close"].pct_change().fillna(0.0)
    return df.groupby("session")["bar_ret"].std()
vol_pub = sess_vol(pub_bars).reindex(pub_sess).to_numpy()
vol_pri = sess_vol(pri_bars).reindex(pri_sess).to_numpy()

# Order test in same session order as a reference submission (ridge_top10.csv)
ref = pd.read_csv(ROOT/"submissions/chatgpt/ridge_top10.csv").sort_values("session").reset_index(drop=True)
ref_sess = ref["session"].values

# Pub sids may be in numeric order already (1000..10999); same for priv. Make lookup
X_test_all = np.zeros((20000, X_pub.shape[1]))
vol_all = np.zeros(20000)
# indices: pub_sess and pri_sess tell us positions
df_pub = pd.DataFrame({"s":pub_sess}); df_pub["X"]=list(X_pub); df_pub["v"]=vol_pub
df_pri = pd.DataFrame({"s":pri_sess}); df_pri["X"]=list(X_pri); df_pri["v"]=vol_pri
both = pd.concat([df_pub,df_pri]).set_index("s")
both = both.reindex(ref_sess)
X_test = np.stack(both["X"].values)
vol_test = both["v"].to_numpy()

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_tr)
Xte_s = scaler.transform(X_test)

# Multiple alphas
for alpha in [30.0, 100.0, 300.0, 1000.0, 3000.0]:
    m = Ridge(alpha=alpha).fit(Xtr_s, y)
    pred = m.predict(Xte_s)
    # shape: thresholded_inv_vol with q=0.35, then shrink_alpha=0.5 short_floor=0.30
    cutoff = np.quantile(np.abs(pred), 0.35)
    pos = pred / np.maximum(vol_test, 1e-6)
    pos[np.abs(pred)<cutoff] = 0.0
    m_abs = np.mean(np.abs(pos)); scaled = pos/m_abs if m_abs>0 else pos
    final = 0.5*scaled + 0.5*1.0
    final = np.maximum(final, 0.30)
    out = pd.DataFrame({"session": ref_sess, "target_position": final})
    nm = f"ridge_mlm_a{int(alpha)}.csv"
    out.to_csv(ROOT/"submissions"/nm, index=False)
    print(f"{nm}: mean={final.mean():.4f} std={final.std():.4f} min={final.min():.3f} max={final.max():.3f}")
print("done")
