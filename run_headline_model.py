"""Per-headline direction model.

Features per headline:
  - template_id one-hot
  - sector one-hot
  - region one-hot
  - bar_ix (raw)
  - log(1+$amount) parsed from headline
  - percentage parsed from headline
  - sentiment signed score (finbert)
  - sentiment label one-hot
  - session-level features: session vol up to bar_ix, session mom up to bar_ix

Target per headline: forward K-bar close/close_at_bar_ix return (K=5).

Train ridge. Predict for each test headline. Aggregate per session via
  sum( pred * recency_weight ) where recency_weight = exp(-(49-bar_ix)/tau)
Then shape to positions via inv-vol threshold + finalize.
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "models"))
from features import extract_event, SENT_MAP, N_TEMPLATES, SECTORS, REGIONS  # type: ignore

DATA = ROOT / "data"
SUB = ROOT / "submissions"

K_HORIZON = 5
RECENCY_TAU = 10.0

_AMOUNT_RE = re.compile(r"\$([\d.]+)([KMBT]?)")
_PCT_RE = re.compile(r"([\d.]+)%")
_UNITS = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_amount(h: str) -> float:
    m = _AMOUNT_RE.search(h)
    if not m:
        return 0.0
    try:
        return float(m.group(1)) * _UNITS.get(m.group(2), 1)
    except Exception:
        return 0.0


def parse_pct(h: str) -> float:
    m = _PCT_RE.search(h)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0


SECTOR_IDX = {s: i for i, s in enumerate(SECTORS)}
REGION_IDX = {r: i for i, r in enumerate(REGIONS)}
N_SEC = len(SECTORS)
N_REG = len(REGIONS)


def build_session_stats(bars: pd.DataFrame) -> pd.DataFrame:
    """For each (session, bar_ix) compute rolling vol & mom up to bar_ix."""
    b = bars.sort_values(["session", "bar_ix"]).copy()
    b["bar_ret"] = b.groupby("session")["close"].pct_change().fillna(0.0)
    g = b.groupby("session")
    b["cum_ret"] = g["bar_ret"].cumsum()
    # rolling std over last 10 bars
    b["roll_vol10"] = g["bar_ret"].rolling(10, min_periods=2).std().reset_index(0, drop=True).fillna(0.0)
    return b[["session", "bar_ix", "cum_ret", "roll_vol10"]]


def featurize(hdf: pd.DataFrame, bars_seen: pd.DataFrame,
              fwd_ret_map: pd.Series | None = None) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Return X (sparse), bar_ix array, session array, y (NaN where no target)."""
    hdf = hdf.copy()
    # sentiment join
    sent = hdf["headline"].map(SENT_MAP["signed"].to_dict()).fillna(0.0).to_numpy()
    lbl = hdf["headline"].map(SENT_MAP["label"].to_dict()).fillna("neutral").to_numpy()
    # event extraction
    triples = [extract_event(h) for h in hdf["headline"]]
    tids = np.array([t[0] for t in triples], dtype=int)
    secs = [t[1] for t in triples]
    regs = [t[2] for t in triples]
    # numeric scalars
    bar_ix = hdf["bar_ix"].to_numpy(dtype=float)
    amt = np.array([parse_amount(h) for h in hdf["headline"]], dtype=float)
    pct = np.array([parse_pct(h) for h in hdf["headline"]], dtype=float)
    sent_label_pos = (lbl == "positive").astype(float)
    sent_label_neg = (lbl == "negative").astype(float)

    # session stats join
    sess_stats = build_session_stats(bars_seen)
    key = pd.DataFrame({"session": hdf["session"].to_numpy(), "bar_ix": hdf["bar_ix"].to_numpy()})
    merged = key.merge(sess_stats, on=["session", "bar_ix"], how="left")
    cum_ret = merged["cum_ret"].fillna(0.0).to_numpy()
    roll_vol10 = merged["roll_vol10"].fillna(0.0).to_numpy()

    # sparse one-hots
    n = len(hdf)
    # template one-hot (tid may be -1 → drop)
    tid_valid = (tids >= 0) & (tids < N_TEMPLATES)
    tid_safe = np.where(tid_valid, tids, 0)
    tid_oh = csr_matrix((np.where(tid_valid, 1.0, 0.0),
                         (np.arange(n), tid_safe)), shape=(n, N_TEMPLATES))
    # sector one-hot
    sec_idx = np.array([SECTOR_IDX.get(s, -1) for s in secs])
    sec_valid = sec_idx >= 0
    sec_safe = np.where(sec_valid, sec_idx, 0)
    sec_oh = csr_matrix((np.where(sec_valid, 1.0, 0.0),
                         (np.arange(n), sec_safe)), shape=(n, N_SEC))
    # region one-hot
    reg_idx = np.array([REGION_IDX.get(r, -1) for r in regs])
    reg_valid = reg_idx >= 0
    reg_safe = np.where(reg_valid, reg_idx, 0)
    reg_oh = csr_matrix((np.where(reg_valid, 1.0, 0.0),
                         (np.arange(n), reg_safe)), shape=(n, N_REG))

    # dense numeric block (will standardize later): bar_ix, log1p(amt), pct/100, sent, sent_pos, sent_neg, cum_ret, roll_vol10
    dense = np.stack([
        bar_ix,
        np.log1p(amt),
        pct / 100.0,
        sent.astype(float),
        sent_label_pos,
        sent_label_neg,
        cum_ret,
        roll_vol10,
    ], axis=1)

    X = hstack([tid_oh, sec_oh, reg_oh, csr_matrix(dense)]).tocsr()

    # compute y (fwd K-bar return from bar_ix) if fwd_ret_map provided
    if fwd_ret_map is None:
        y = np.full(n, np.nan)
    else:
        y = fwd_ret_map.reindex(list(zip(hdf["session"].to_numpy(), hdf["bar_ix"].to_numpy()))).to_numpy()
        y = np.where(np.isfinite(y), y, 0.0)  # NaN → 0 so ridge can handle; we'll mask in fit

    return X, bar_ix, hdf["session"].to_numpy(), y


def compute_fwd_returns(bars: pd.DataFrame, K: int) -> pd.Series:
    """For each (session, bar_ix) compute close[bar_ix+K]/close[bar_ix] - 1,
    clipped at session end. Returns a Series indexed by (session, bar_ix)."""
    b = bars.sort_values(["session", "bar_ix"]).copy()
    b["close_shift"] = b.groupby("session")["close"].shift(-K)
    b["close_last"] = b.groupby("session")["close"].transform("last")
    b["close_fwd"] = b["close_shift"].fillna(b["close_last"])
    b["fwd_ret"] = b["close_fwd"] / b["close"] - 1
    s = b.set_index(["session", "bar_ix"])["fwd_ret"]
    return s


# Load bars (train: seen + unseen concatenated for horizon coverage)
train_seen = pd.read_parquet(DATA / "bars_seen_train.parquet")
train_unseen = pd.read_parquet(DATA / "bars_unseen_train.parquet")
train_bars = pd.concat([train_seen, train_unseen], ignore_index=True)
pub_bars = pd.read_parquet(DATA / "bars_seen_public_test.parquet")
pri_bars = pd.read_parquet(DATA / "bars_seen_private_test.parquet")

# Headlines
train_h = pd.read_parquet(DATA / "headlines_seen_train.parquet")
pub_h = pd.read_parquet(DATA / "headlines_seen_public_test.parquet")
pri_h = pd.read_parquet(DATA / "headlines_seen_private_test.parquet")

# Train fwd returns
train_fwd = compute_fwd_returns(train_bars, K_HORIZON)
# Test fwd returns only computable within seen bars (0..49), no extension to unseen.
# We still compute them for the test headlines' bar_ix using the seen test bars only —
# but those are not the model's target (we use trained weights), so we don't need fwd for test.
# Featurize
print("featurizing train...")
Xtr, bar_tr, sess_tr, ytr = featurize(train_h, train_seen, train_fwd)
print("train matrix:", Xtr.shape, "valid y:", np.isfinite(ytr).sum())

print("featurizing public test...")
Xpub, bar_pub, sess_pub, _ = featurize(pub_h, pub_bars)
print("pub matrix:", Xpub.shape)
print("featurizing private test...")
Xpri, bar_pri, sess_pri, _ = featurize(pri_h, pri_bars)
print("pri matrix:", Xpri.shape)

# standardize the final 8 dense cols (which are last 8 in the stacked matrix)
# We'll convert to dense-partial for standardization of those trailing cols.
n_onehot = N_TEMPLATES + N_SEC + N_REG
print("n_onehot:", n_onehot, "total features:", Xtr.shape[1])

def zscore_trailing(X, mu, sd, n_trail):
    X = X.toarray() if hasattr(X, "toarray") else X
    X = X.astype(np.float64, copy=True)
    X[:, -n_trail:] = (X[:, -n_trail:] - mu) / np.where(sd < 1e-8, 1.0, sd)
    return X

mu = np.asarray(Xtr[:, -8:].mean(axis=0)).ravel()
sd = np.asarray(np.sqrt(((Xtr[:, -8:].toarray() - mu) ** 2).mean(axis=0))).ravel()
Xtr_d = zscore_trailing(Xtr, mu, sd, 8)
Xpub_d = zscore_trailing(Xpub, mu, sd, 8)
Xpri_d = zscore_trailing(Xpri, mu, sd, 8)

# Fit ridge on train headlines
for alpha in [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]:
    m = Ridge(alpha=alpha).fit(Xtr_d, ytr)
    pred_pub = m.predict(Xpub_d)
    pred_pri = m.predict(Xpri_d)
    # aggregate per session: sum( pred * recency )
    def agg(pred, bar, sess):
        rec = np.exp(-(49.0 - bar) / RECENCY_TAU)
        df = pd.DataFrame({"session": sess, "score": pred * rec})
        return df.groupby("session")["score"].sum()
    s_pub = agg(pred_pub, bar_pub, sess_pub)
    s_pri = agg(pred_pri, bar_pri, sess_pri)

    # Build a positions Series indexed by all test sessions (1000..20999)
    ref = pd.read_csv(ROOT / "submissions/chatgpt/ridge_top10.csv").sort_values("session").reset_index(drop=True)
    sess_all = ref["session"].values
    score = pd.Series(0.0, index=sess_all)
    score.loc[s_pub.index] = s_pub.values
    score.loc[s_pri.index] = s_pri.values

    # vol per session
    def sess_vol(bars):
        b = bars.sort_values(["session","bar_ix"]).copy()
        b["bar_ret"] = b.groupby("session")["close"].pct_change().fillna(0.0)
        return b.groupby("session")["bar_ret"].std()
    vol = pd.concat([sess_vol(pub_bars), sess_vol(pri_bars)]).reindex(sess_all).to_numpy()

    pred = score.to_numpy()
    cutoff = np.quantile(np.abs(pred), 0.35)
    pos = pred / np.maximum(vol, 1e-6)
    pos[np.abs(pred) < cutoff] = 0.0
    m_abs = np.mean(np.abs(pos)); scaled = pos / m_abs if m_abs > 0 else pos
    final = np.maximum(0.5 * scaled + 0.5, 0.30)
    out = pd.DataFrame({"session": sess_all, "target_position": final})
    nm = f"ridge_hl_a{int(alpha)}.csv"
    out.to_csv(SUB / nm, index=False)
    print(f"{nm:30s} mean={final.mean():.4f} std={final.std():.4f} min={final.min():.3f} max={final.max():.3f}")
