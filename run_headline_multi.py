"""Multi-horizon headline-level ridge models for blending.

Variants:
  - hl_k3:  fwd 3-bar return target
  - hl_k10: fwd 10-bar return target
  - hl_end: close_at_session_end / close_at_bar_ix - 1 (train: close[99]; horizon crosses seen/unseen)

All share the same feature pipeline from run_headline_model.py.
"""
from pathlib import Path
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from run_headline_model import (  # type: ignore
    featurize, build_session_stats, compute_fwd_returns, RECENCY_TAU,
    DATA, SUB,
)

train_seen = pd.read_parquet(DATA/"bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA/"bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA/"bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA/"bars_seen_private_test.parquet")

train_h = pd.read_parquet(DATA/"headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA/"headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA/"headlines_seen_private_test.parquet")

def session_end_target(bars_all: pd.DataFrame) -> pd.Series:
    """(close at last bar of session) / close[bar_ix] - 1, indexed by (session, bar_ix)."""
    b = bars_all.sort_values(["session","bar_ix"]).copy()
    last_close = b.groupby("session")["close"].transform("last")
    b["fwd_ret"] = last_close / b["close"] - 1
    return b.set_index(["session","bar_ix"])["fwd_ret"]

def featurize_and_predict(train_target_map: pd.Series, alpha: float, name: str):
    print(f"--- {name} alpha={alpha} ---")
    Xtr, bar_tr, sess_tr, ytr = featurize(train_h, train_seen, train_target_map)
    Xpub, bar_pub, sess_pub, _ = featurize(pub_h, pub_bars)
    Xpri, bar_pri, sess_pri, _ = featurize(pri_h, pri_bars)
    mu = np.asarray(Xtr[:, -8:].mean(axis=0)).ravel()
    sd = np.asarray(np.sqrt(((Xtr[:, -8:].toarray() - mu) ** 2).mean(axis=0))).ravel()
    def zs(X):
        X = X.toarray().astype(np.float64, copy=True)
        X[:, -8:] = (X[:, -8:] - mu) / np.where(sd < 1e-8, 1.0, sd)
        return X
    Xtr_d = zs(Xtr); Xpub_d = zs(Xpub); Xpri_d = zs(Xpri)
    m = Ridge(alpha=alpha).fit(Xtr_d, ytr)
    pred_pub = m.predict(Xpub_d)
    pred_pri = m.predict(Xpri_d)
    def agg(pred, bar, sess):
        rec = np.exp(-(49.0 - bar) / RECENCY_TAU)
        return pd.DataFrame({"s":sess,"v":pred*rec}).groupby("s")["v"].sum()
    s_pub = agg(pred_pub, bar_pub, sess_pub); s_pri = agg(pred_pri, bar_pri, sess_pri)
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
    pd.DataFrame({"session":sess_all,"target_position":final}).to_csv(SUB/f"{name}.csv",index=False)
    print(f"{name}: mean={final.mean():.4f} std={final.std():.4f} min={final.min():.3f} max={final.max():.3f}")

# K=3 target
featurize_and_predict(compute_fwd_returns(train_bars, 3), 3000.0, "ridge_hl_k3_a3000")
# K=10 target
featurize_and_predict(compute_fwd_returns(train_bars, 10), 3000.0, "ridge_hl_k10_a3000")
# session-end target (horizon-to-close[99])
featurize_and_predict(session_end_target(train_bars), 3000.0, "ridge_hl_end_a3000")
