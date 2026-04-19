"""Shared feature builders + scoring helpers for the catboost_bars pipeline."""
from __future__ import annotations
import re
from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SEED = 42
N_BOOTSTRAP = 2000

# ---------- sentiment lookup (shared) ----------
_finbert = pd.read_parquet(DATA_DIR / "headlines_finbert_sentiment.parquet")
_finbert["signed"] = np.where(
    _finbert["label"] == "positive", _finbert["score"],
    np.where(_finbert["label"] == "negative", -_finbert["score"], 0.0)
)
SENT_MAP = _finbert.set_index("headline")[["label", "signed"]]


# ---------- template / sector / region extractors (derived from headline_features.csv) ----------
_hf = pd.read_csv(MODELS_DIR / "headline_features.csv", usecols=["template_index", "template_text"])
_tpl_pairs = [(int(i), t) for i, t in _hf.drop_duplicates().sort_values("template_index").values.tolist()
              if int(i) >= 0]
SECTORS = sorted(c[len("sector_"):] for c in pd.read_csv(MODELS_DIR / "headline_features.csv", nrows=0).columns
                 if c.startswith("sector_") and c != "sector_<NONE>")
REGIONS = sorted(c[len("region_"):] for c in pd.read_csv(MODELS_DIR / "headline_features.csv", nrows=0).columns
                 if c.startswith("region_") and c != "region_<NONE>")
N_TEMPLATES = max(i for i, _ in _tpl_pairs) + 1

# Authoritative (headline -> template_index) lookup from train/public/private CSVs
# when available. Any unseen headline still falls back to the rule-based parser.
_lookup_dfs = [pd.read_csv(MODELS_DIR / "headline_features.csv", usecols=["title", "template_index"])]
for _path in (
    MODELS_DIR / "headline_features_public.csv",
    MODELS_DIR / "headline_features_private.csv",
):
    if _path.exists():
        _lookup_dfs.append(pd.read_csv(_path, usecols=["title", "template_index"]))
_lookup_df = pd.concat(_lookup_dfs, ignore_index=True).drop_duplicates("title")
TEMPLATE_LOOKUP: dict[str, int] = dict(zip(_lookup_df["title"], _lookup_df["template_index"].astype(int)))


def _compile_template(text: str) -> re.Pattern:
    """Convert a template like '<COMPANY> secures $X contract with <PARTNER>'
    into a regex that matches actual headlines (fullmatch)."""
    # Protect placeholders before escaping, then restore as regex pieces.
    markers = {
        "<COMPANY>": "\x00COMP\x00",
        "<PARTNER>": "\x00PART\x00",
        "<SECTOR>": "\x00SECT\x00",
        "<REGION>": "\x00REGN\x00",
    }
    tmp = text
    for k, v in markers.items():
        tmp = tmp.replace(k, v)
    tmp = re.escape(tmp)
    tmp = tmp.replace("\x00COMP\x00", r".+?")
    tmp = tmp.replace("\x00PART\x00", r".+?")
    tmp = tmp.replace("\x00SECT\x00", r"(?P<sector>[^,]+?)")
    tmp = tmp.replace("\x00REGN\x00", r"(?P<region>[^,]+?)")
    # literal "$X" and "N%" inside template become actual amounts in headlines
    tmp = tmp.replace(r"\$X", r"\$[\d.]+[KMBT]?")
    tmp = tmp.replace(r"N%", r"[\d.]+%")
    return re.compile(tmp)


_COMPILED = [(idx, _compile_template(txt)) for idx, txt in _tpl_pairs]
# longest template first so specific matches beat generic prefixes
_COMPILED.sort(key=lambda t: -len(t[1].pattern))
_SECTOR_SET = set(SECTORS)
_REGION_SET = set(REGIONS)


from template_parser import parse_template_index


@lru_cache(maxsize=None)
def extract_event(headline: str) -> tuple[int, str, str]:
    """Return (template_id, sector, region). sector/region are '' if not applicable.
    Template id resolution order:
      1. CSV lookup (100% accurate for train + public test headlines)
      2. Rule-based parser (canonical bucketing for any unseen headline)
      3. Regex fallback (for edge cases the parser misses)
    """
    tid = TEMPLATE_LOOKUP.get(headline)
    if tid is None:
        tid = parse_template_index(headline)
    # Still run regex to pull sector/region groups out of the headline.
    for _, pat in _COMPILED:
        m = pat.fullmatch(headline)
        if m:
            gd = m.groupdict()
            sec = gd.get("sector") or ""
            reg = gd.get("region") or ""
            sec = sec if sec in _SECTOR_SET else ""
            reg = reg if reg in _REGION_SET else ""
            return tid, sec, reg
    return tid, "", ""


RECENCY_TAU = 10.0  # exp decay half-scale for recency weighting on bar_ix


def _attach_tid(hdf: pd.DataFrame) -> pd.DataFrame:
    """Attach template_id column via regex extractor."""
    hdf = hdf.copy()
    hdf["_tid"] = [extract_event(h)[0] for h in hdf["headline"]]
    return hdf


def fit_template_impacts(headlines_df: pd.DataFrame, bars_df: pd.DataFrame,
                         K: int = 5, prior_n: float = 30.0) -> np.ndarray:
    """Per-template impact = shrunk mean of the forward K-bar return of headlines
    matching that template. Uses the headline's local effect (bar_ix..bar_ix+K),
    not session-long return, so drift + other events don't muddy the estimate.

    Shrinkage toward the global headline mean with strength `prior_n` stabilizes
    rare templates: impact = (n*sample_mean + prior_n*global_mean) / (n + prior_n).
    """
    bars = bars_df[["session", "bar_ix", "close"]].set_index(["session", "bar_ix"])["close"]
    max_bar = bars_df.groupby("session")["bar_ix"].max()
    h = _attach_tid(headlines_df)
    h = h[h["_tid"] >= 0].copy()
    h["c0"] = bars.reindex(list(zip(h["session"], h["bar_ix"]))).values
    end = np.minimum(h["bar_ix"].to_numpy() + K,
                     h["session"].map(max_bar).to_numpy())
    h["c1"] = bars.reindex(list(zip(h["session"], end))).values
    h["fwd_ret"] = h["c1"] / h["c0"] - 1
    h = h.dropna(subset=["fwd_ret"])
    global_mean = float(h["fwd_ret"].mean())
    per = h.groupby("_tid")["fwd_ret"].agg(["mean", "count"])
    impacts = np.full(N_TEMPLATES, global_mean, dtype=float)
    tids = np.asarray(per.index.to_numpy(), dtype=int)
    means = per["mean"].to_numpy(dtype=float)
    counts = per["count"].to_numpy(dtype=float)
    impacts[tids] = (counts * means + prior_n * global_mean) / (counts + prior_n)
    return impacts


HORIZONS = (3, 5, 10)  # forward-bar windows to estimate template impact at


def _feat_cols(horizons=HORIZONS) -> list[str]:
    cols = []
    for K in horizons:
        cols += [f"event_impact_k{K}", f"event_impact_recent_k{K}"]
    return cols


def fit_template_impacts_multi(headlines_df: pd.DataFrame, bars_df: pd.DataFrame,
                               horizons=HORIZONS) -> dict[int, np.ndarray]:
    """Per-horizon impact tables: {K: array[N_TEMPLATES]}."""
    return {K: fit_template_impacts(headlines_df, bars_df, K=K) for K in horizons}


def build_event_features_multi(hdf: pd.DataFrame, all_sessions: np.ndarray,
                               impacts_by_k: dict[int, np.ndarray]) -> pd.DataFrame:
    """Produces 2 cols per horizon: event_impact_k{K}, event_impact_recent_k{K}."""
    dfs = []
    for K, imp in impacts_by_k.items():
        d = build_event_features(hdf, all_sessions, imp)
        d.columns = [f"event_impact_k{K}", f"event_impact_recent_k{K}"]
        dfs.append(d)
    return pd.concat(dfs, axis=1)


def build_event_features_oof(headlines_df: pd.DataFrame, bars_df: pd.DataFrame,
                             sessions: np.ndarray,
                             n_splits: int = 5, seed: int = SEED,
                             horizons=HORIZONS) -> pd.DataFrame:
    """OOF version for TRAIN features: each session's event-impact cols are
    computed from impacts fit on a fold that EXCLUDES the session. Outputs
    2 cols per horizon (sum + recency-weighted sum)."""
    from sklearn.model_selection import KFold
    cols = _feat_cols(horizons)
    out = pd.DataFrame(0.0, index=sessions, columns=cols)
    out.index.name = "session"
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(sessions):
        tr_s, va_s = sessions[tr_idx], sessions[va_idx]
        h_tr = headlines_df[headlines_df["session"].isin(tr_s)]
        b_tr = bars_df[bars_df["session"].isin(tr_s)]
        impacts_k = fit_template_impacts_multi(h_tr, b_tr, horizons=horizons)
        h_va = headlines_df[headlines_df["session"].isin(va_s)]
        va_features = build_event_features_multi(h_va, va_s, impacts_k)
        out.loc[va_s, cols] = va_features[cols].values
    return out.sort_index()


def build_event_features(hdf: pd.DataFrame, all_sessions: np.ndarray,
                         impacts: np.ndarray) -> pd.DataFrame:
    """Per-session event-impact features from template_id + bar_ix + impact table.
      event_impact          = sum of impacts across session's headlines
      event_impact_recent   = sum weighted by exp(-(49 - bar_ix)/RECENCY_TAU)
    Works for train and test (same impact table)."""
    h = _attach_tid(hdf)
    h = h[h["_tid"] >= 0].copy()
    h["impact"] = impacts[h["_tid"].to_numpy()]
    h["recency_w"] = np.exp(-(49.0 - h["bar_ix"].to_numpy()) / RECENCY_TAU)
    h["impact_recent"] = h["impact"] * h["recency_w"]
    agg = h.groupby("session")[["impact", "impact_recent"]].sum()
    out = pd.DataFrame(index=all_sessions, columns=["event_impact", "event_impact_recent"],
                       dtype=float).fillna(0.0)
    out.index.name = "session"
    out.loc[agg.index, "event_impact"] = agg["impact"].values
    out.loc[agg.index, "event_impact_recent"] = agg["impact_recent"].values
    return out.sort_index()


# ---------- feature builders ----------
def build_bar_features(seen_bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 50 seen bars per session into a flat feature row."""
    df = seen_bars.sort_values(["session", "bar_ix"]).copy()
    df["bar_ret"] = df.groupby("session")["close"].pct_change().fillna(0.0)
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["close_pos"] = (df["close"] - df["low"]) / df["range"].replace(0, np.nan)

    g = df.groupby("session")
    feats = pd.DataFrame(index=g.groups.keys())
    feats.index.name = "session"

    feats["close_first"] = g["close"].first()
    feats["close_last"] = g["close"].last()
    feats["seen_ret"] = feats["close_last"] / feats["close_first"] - 1
    for n in (1, 3, 5, 10):
        feats[f"mom_{n}"] = df.groupby("session").tail(n).groupby("session")["bar_ret"].sum()
    feats["vol"] = g["bar_ret"].std()
    feats["vol_recent"] = df.groupby("session").tail(10).groupby("session")["bar_ret"].std()
    feats["max_high"] = g["high"].max()
    feats["min_low"] = g["low"].min()
    feats["dist_to_high"] = feats["max_high"] / feats["close_last"] - 1
    feats["dist_to_low"] = feats["min_low"] / feats["close_last"] - 1
    feats["body_mean"] = g["body"].mean()
    feats["range_mean"] = g["range"].mean()
    feats["close_pos_mean"] = g["close_pos"].mean()
    feats["close_pos_last"] = df.groupby("session")["close_pos"].last()
    cummax = g["close"].cummax()
    dd = df["close"] / cummax - 1
    feats["max_drawdown"] = dd.groupby(df["session"]).min()
    feats["ret_skew"] = g["bar_ret"].skew()

    return feats.replace([np.inf, -np.inf], np.nan).sort_index()


def build_headline_features(hdf: pd.DataFrame, all_sessions: np.ndarray) -> pd.DataFrame:
    """Per-session headline features: counts, timing, FinBERT sentiment aggregates.
    Works identically for train and test (only needs {session, headline, bar_ix})."""
    hdf = hdf.merge(SENT_MAP, left_on="headline", right_index=True, how="left")
    hdf["signed"] = hdf["signed"].fillna(0.0)
    hdf["is_pos"] = (hdf["label"] == "positive").astype(int)
    hdf["is_neg"] = (hdf["label"] == "negative").astype(int)
    hdf["recent"] = (hdf["bar_ix"] >= 40).astype(int)
    g = hdf.groupby("session")
    out = pd.DataFrame(index=all_sessions)
    out.index.name = "session"
    out["hl_n"] = g.size()
    out["hl_n_recent"] = g["recent"].sum()
    out["hl_last_bar"] = g["bar_ix"].max()
    out["hl_mean_bar"] = g["bar_ix"].mean()
    out["hl_net_sent"] = g["signed"].sum()
    out["hl_net_sent_recent"] = hdf[hdf["recent"] == 1].groupby("session")["signed"].sum()
    out["hl_mean_sent"] = g["signed"].mean()
    out["hl_n_pos"] = g["is_pos"].sum()
    out["hl_n_neg"] = g["is_neg"].sum()
    for c in ["hl_n", "hl_n_recent", "hl_net_sent", "hl_net_sent_recent",
              "hl_mean_sent", "hl_n_pos", "hl_n_neg"]:
        out[c] = out[c].fillna(0.0)
    out["hl_last_bar"] = out["hl_last_bar"].fillna(-1)
    out["hl_mean_bar"] = out["hl_mean_bar"].fillna(-1)
    return out.sort_index()


# ---------- data loading wrappers ----------
def load_train_base() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """X with bars+sentiment (no event-impact cols), y, raw headlines df, and
    the full seen+unseen bars df (needed for forward-return impact fitting).
    Used by evaluate.py so each split can refit impacts without leakage."""
    seen = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
    unseen = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
    # seen is bar_ix 0..49, unseen is 50..99 — concat gives a full 0..99 window
    bars_full = pd.concat([seen, unseen], ignore_index=True)
    headlines = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")
    last_seen = seen.sort_values("bar_ix").groupby("session")["close"].last()
    last_unseen = unseen.sort_values("bar_ix").groupby("session")["close"].last()
    target = (last_unseen / last_seen - 1).rename("y")
    X = build_bar_features(seen)
    X = X.join(build_headline_features(headlines, X.index.to_numpy()))
    y = target.reindex(X.index)
    assert y.notna().all(), "target has NaNs"
    return X, y, headlines, bars_full


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    """X with bars + sentiment + OOF event-impact cols (no target leakage)."""
    X, y, headlines, bars_full = load_train_base()
    X = X.join(build_event_features_oof(headlines, bars_full, X.index.to_numpy()))
    return X, y


def load_test(impacts: dict[int, np.ndarray] | None = None) -> pd.DataFrame:
    """If impacts is None, fit on full train headlines+bars (for submission time)."""
    test_seen = pd.concat([
        pd.read_parquet(DATA_DIR / "bars_seen_public_test.parquet"),
        pd.read_parquet(DATA_DIR / "bars_seen_private_test.parquet"),
    ], ignore_index=True)
    test_headlines = pd.concat([
        pd.read_parquet(DATA_DIR / "headlines_seen_public_test.parquet"),
        pd.read_parquet(DATA_DIR / "headlines_seen_private_test.parquet"),
    ], ignore_index=True)
    if impacts is None:
        _, _, train_headlines, train_bars = load_train_base()
        impacts = fit_template_impacts_multi(train_headlines, train_bars)
    X = build_bar_features(test_seen)
    X = X.join(build_headline_features(test_headlines, X.index.to_numpy()))
    X = X.join(build_event_features_multi(test_headlines, X.index.to_numpy(), impacts))
    return X


# ---------- scoring + position shaping ----------
def sharpe(positions: np.ndarray, returns: np.ndarray) -> float:
    pnl = positions * returns
    return float(pnl.mean() / pnl.std() * 16) if pnl.std() > 0 else 0.0


def sharpe_bootstrap_ci(positions: np.ndarray, returns: np.ndarray,
                        n: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    N = len(positions)
    pnl = positions * returns
    boots = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N)
        p = pnl[idx]
        s = p.std()
        boots[i] = (p.mean() / s * 16) if s > 0 else 0.0
    return sharpe(positions, returns), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def shape_positions(pred: np.ndarray, vol: np.ndarray, kind: str,
                    threshold_q: float = 0.25) -> np.ndarray:
    """Turn raw predictions into trading positions. Sharpe is scale-invariant so
    only the relative cross-session weighting + sign matter."""
    if kind == "raw":
        return pred
    if kind == "sign":
        return np.sign(pred)
    if kind == "inv_vol":
        return pred / np.maximum(vol, 1e-6)
    if kind == "thresholded":
        cutoff = np.quantile(np.abs(pred), threshold_q)
        out = pred.copy()
        out[np.abs(out) < cutoff] = 0.0
        return out
    if kind == "thresholded_inv_vol":
        cutoff = np.quantile(np.abs(pred), threshold_q)
        out = pred / np.maximum(vol, 1e-6)
        out[np.abs(pred) < cutoff] = 0.0
        return out
    raise ValueError(kind)


# Shrink toward constant-long and floor shorts at 0. The drift is the bulk of the
# score; the model tilt is small and noisy, so we lean on the guaranteed-good
# long bet and use the model only to modulate size.
SHRINK_ALPHA = 0.7
SHORT_FLOOR = 0.0


def finalize(pos: np.ndarray,
             shrink_alpha: float = SHRINK_ALPHA,
             short_floor: float = SHORT_FLOOR) -> np.ndarray:
    m = np.mean(np.abs(pos))
    scaled = pos / m if m > 0 else pos
    blended = shrink_alpha * scaled + (1 - shrink_alpha) * 1.0
    return np.maximum(blended, short_floor)
