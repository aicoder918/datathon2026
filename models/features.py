"""Feature builder for the Sharpe-optimal position model.

Produces exactly 15 features per session from (bars, headlines, finbert-sentiment):
  10 price-path features on bars with bar_ix <= SEEN_MAX
   5 headline features on headlines with bar_ix <= SEEN_MAX

All features are computed per-session in isolation. No train-wide aggregates,
no forward-looking columns, no one-hots. See plan at
~/.claude/plans/ok-lets-reason-from-zany-sprout.md for motivation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SEEN_MAX = 49          # last bar_ix we are allowed to look at
HALFWAY_BAR = 49       # close at which the position is taken
END_BAR = 99           # close at which the position is unwound (train only)
LATE_START = 40        # first bar_ix considered "late" for headline recency

PRICE_FEATURES = [
    "ret_total", "ret_last5", "ret_last10", "ret_last20",
    "rv", "rv_late_ratio", "rv_last10", "pk_vol",
    "trend_slope_norm", "drawdown", "acf1", "body_frac",
]
HEADLINE_FEATURES = [
    "n_headlines", "sent_sum", "sent_recent_w", "sent_exp_w",
    "sent_last10", "sent_neg_sum", "last_hl_signed", "last_hl_age",
    "sent_change", "neg_frac",
]
# Template-impact features are added by template_impact.py *after* the per-fold
# OOF impact table is computed; they're appended to the FEATURE_COLS list at runtime.
TEMPLATE_FEATURES = ["tpl_late_score", "tpl_all_score"]
FEATURE_COLS = PRICE_FEATURES + HEADLINE_FEATURES + TEMPLATE_FEATURES

VOL_FEATURES = ["rv", "pk_vol", "rv_late_ratio", "rv_last10",
                "n_headlines", "sent_last10"]

EXP_DECAY_TAU = 10.0   # sentiment exp-decay time constant (in bars)
EARLY_END = 30         # last bar_ix considered "early" for sent_change


# ---------------------------------------------------------------- sentiment

def load_finbert(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return a DataFrame with columns [headline, signed] where signed = sign * score."""
    s = pd.read_parquet(data_dir / "headlines_finbert_sentiment.parquet")
    sign = np.where(s["label"] == "positive", 1.0,
                    np.where(s["label"] == "negative", -1.0, 0.0))
    out = pd.DataFrame({
        "headline": s["headline"].values,
        "signed": sign * s["score"].values,
    })
    return out.drop_duplicates("headline")


# ---------------------------------------------------------------- price features

def _price_features_for_session(g: pd.DataFrame) -> dict:
    """g is bars for one session, sorted by bar_ix, restricted to bar_ix <= SEEN_MAX."""
    c = g["close"].to_numpy(dtype=np.float64)
    o = g["open"].to_numpy(dtype=np.float64)
    h = g["high"].to_numpy(dtype=np.float64)
    lo = g["low"].to_numpy(dtype=np.float64)
    n = len(c)

    logc = np.log(c)
    dlog = np.diff(logc)  # length n-1

    ret_total = logc[-1] - logc[0] if n >= 2 else 0.0
    ret_last5 = logc[-1] - logc[-6] if n >= 6 else ret_total
    ret_last10 = logc[-1] - logc[-11] if n >= 11 else ret_total
    ret_last20 = logc[-1] - logc[-21] if n >= 21 else ret_total

    rv = float(dlog.std(ddof=1)) if n >= 3 else 0.0
    if n >= 32:
        rv_early = dlog[:29].std(ddof=1)  # returns over bars 1..29
        rv_late = dlog[29:].std(ddof=1)   # returns over bars 30..49
        rv_late_ratio = float(rv_late / rv_early) if rv_early > 1e-12 else 1.0
        rv_last10 = float(dlog[-10:].std(ddof=1))
    else:
        rv_late_ratio = 1.0
        rv_last10 = rv

    # Parkinson volatility: sum((ln(h/l))^2) / (4*ln2*n), root
    hl = np.log(np.maximum(h, 1e-12) / np.maximum(lo, 1e-12))
    pk_var = float((hl ** 2).sum() / (4.0 * np.log(2.0) * max(n, 1)))
    pk_vol = float(np.sqrt(max(pk_var, 0.0)))

    # Trend slope of log(close) vs bar_ix, scaled by rv
    if n >= 3:
        x = np.arange(n, dtype=np.float64)
        xm = x - x.mean()
        ym = logc - logc.mean()
        denom = (xm * xm).sum()
        slope = float((xm * ym).sum() / denom) if denom > 0 else 0.0
        trend_slope_norm = slope / rv if rv > 1e-12 else 0.0
    else:
        trend_slope_norm = 0.0

    running_max = np.maximum.accumulate(c)
    drawdown = float((c[-1] - running_max[-1]) / running_max[-1])  # 0 if at peak, <0 otherwise

    if len(dlog) >= 3:
        d_mean = dlog.mean()
        d_centered = dlog - d_mean
        denom = (d_centered * d_centered).sum()
        acf1 = float((d_centered[:-1] * d_centered[1:]).sum() / denom) if denom > 1e-18 else 0.0
    else:
        acf1 = 0.0

    hl_range = h - lo
    safe_range = np.where(hl_range > 1e-12, hl_range, 1.0)
    body = np.abs(c - o) / safe_range
    body_frac = float(body.mean()) if n > 0 else 0.0

    return {
        "ret_total": ret_total,
        "ret_last5": ret_last5,
        "ret_last10": ret_last10,
        "ret_last20": ret_last20,
        "rv": rv,
        "rv_late_ratio": rv_late_ratio,
        "rv_last10": rv_last10,
        "pk_vol": pk_vol,
        "trend_slope_norm": trend_slope_norm,
        "drawdown": drawdown,
        "acf1": acf1,
        "body_frac": body_frac,
    }


def build_price_features(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars[bars["bar_ix"] <= SEEN_MAX].sort_values(["session", "bar_ix"])
    rows = []
    for sess, g in bars.groupby("session", sort=True):
        feats = _price_features_for_session(g)
        feats["session"] = int(sess)
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("session")
    return df[PRICE_FEATURES]


# ---------------------------------------------------------------- headline features

def build_headline_features(
    headlines: pd.DataFrame,
    sessions: np.ndarray,
    finbert: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per session in `sessions`, with 5 headline features.

    Missing sessions (no headlines) get zeros.
    """
    if finbert is None:
        finbert = load_finbert()

    h = headlines[headlines["bar_ix"] <= SEEN_MAX].copy()
    h = h.merge(finbert, on="headline", how="left")
    h["signed"] = h["signed"].fillna(0.0)
    bar_ix_f = h["bar_ix"].astype(float)

    h["w_recent"] = (bar_ix_f + 1.0) / float(SEEN_MAX + 1)
    h["w_exp"] = np.exp(-(SEEN_MAX - bar_ix_f) / EXP_DECAY_TAU)
    h["is_neg"] = (h["signed"] < 0).astype(float)
    h["is_late"] = (bar_ix_f >= LATE_START).astype(float)
    h["is_early"] = (bar_ix_f <= EARLY_END).astype(float)
    h["signed_neg"] = np.where(h["signed"] < 0, h["signed"], 0.0)

    h["sent_weighted"] = h["signed"] * h["w_recent"]
    h["sent_exp"] = h["signed"] * h["w_exp"]
    h["sent_late"] = h["signed"] * h["is_late"]
    h["sent_early"] = h["signed"] * h["is_early"]

    # Last-headline features (per session)
    h_sorted = h.sort_values(["session", "bar_ix"])
    last_per = h_sorted.groupby("session").tail(1).set_index("session")
    last_signed = last_per["signed"].rename("last_hl_signed")
    last_age = ((SEEN_MAX - last_per["bar_ix"]).astype(float) / float(SEEN_MAX + 1)
                ).rename("last_hl_age")

    agg = h.groupby("session").agg(
        n_headlines=("headline", "size"),
        sent_sum=("signed", "sum"),
        sent_recent_w=("sent_weighted", "sum"),
        sent_exp_w=("sent_exp", "sum"),
        sent_last10=("sent_late", "sum"),
        sent_early_sum=("sent_early", "sum"),
        sent_neg_sum=("signed_neg", "sum"),
        neg_count=("is_neg", "sum"),
    )
    agg["sent_change"] = agg["sent_last10"] - agg["sent_early_sum"]
    agg = agg.join(last_signed, how="left").join(last_age, how="left")
    agg["neg_frac"] = agg["neg_count"] / agg["n_headlines"].where(agg["n_headlines"] > 0, 1)

    # Reindex to all sessions; missing -> sensible defaults
    agg = agg.reindex(sessions)
    agg["last_hl_age"] = agg["last_hl_age"].fillna(1.0)        # "stalest possible"
    agg["last_hl_signed"] = agg["last_hl_signed"].fillna(0.0)
    for c in agg.columns:
        if c not in ("last_hl_age",):
            agg[c] = agg[c].fillna(0.0)
    return agg[HEADLINE_FEATURES]


# ---------------------------------------------------------------- relevance features

def extract_entity(headline: str) -> str:
    """Primary entity proxy = first 2 capitalized tokens of the headline.
    Headlines in this synthetic dataset always start with the company name
    (e.g. 'Relvon Fuels secures $50M contract...' -> 'Relvon Fuels').
    """
    if not isinstance(headline, str):
        return ""
    parts = headline.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")


def build_relevance_features(
    headlines: pd.DataFrame,
    sessions: np.ndarray,
    finbert: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-session features about the session's primary company.

    Primary company = entity with the most seen-headline mentions in this session.
    Ties broken by latest occurrence (more recent = more likely the focus stock).
    """
    if finbert is None:
        finbert = load_finbert()

    h = headlines[headlines["bar_ix"] <= SEEN_MAX].copy()
    h["entity"] = h["headline"].apply(extract_entity)
    h = h.merge(finbert, on="headline", how="left")
    h["signed"] = h["signed"].fillna(0.0)
    bar_ix_f = h["bar_ix"].astype(float)
    h["w_recent"] = (bar_ix_f + 1.0) / float(SEEN_MAX + 1)
    h["is_late"] = (bar_ix_f >= LATE_START).astype(float)
    h["signed_neg"] = np.where(h["signed"] < 0, h["signed"], 0.0)

    # Per-session primary entity: most frequent, tie-break by latest bar_ix
    h_sorted = h.sort_values(["session", "bar_ix"])
    counts = (h_sorted.groupby(["session", "entity"])
              .agg(n=("entity", "size"), last_bar=("bar_ix", "max"))
              .reset_index())
    counts = counts.sort_values(["session", "n", "last_bar"],
                                ascending=[True, False, False])
    primary = counts.drop_duplicates("session", keep="first")[["session", "entity"]] \
        .rename(columns={"entity": "primary_entity"}).set_index("session")

    h = h.merge(primary, left_on="session", right_index=True, how="left")
    h["is_primary"] = (h["entity"] == h["primary_entity"]).astype(float)
    h["primary_signed"] = h["signed"] * h["is_primary"]
    h["primary_signed_recent"] = h["primary_signed"] * h["w_recent"]
    h["primary_signed_late"] = h["primary_signed"] * h["is_late"]
    h["primary_signed_neg"] = h["signed_neg"] * h["is_primary"]
    h["off_signed"] = h["signed"] * (1.0 - h["is_primary"])

    agg = h.groupby("session").agg(
        n_primary_headlines=("is_primary", "sum"),
        primary_sent_sum=("primary_signed", "sum"),
        primary_sent_recent_w=("primary_signed_recent", "sum"),
        primary_sent_last10=("primary_signed_late", "sum"),
        primary_neg_sum=("primary_signed_neg", "sum"),
        off_sent_sum=("off_signed", "sum"),
        n_distinct_entities=("entity", "nunique"),
        n_total=("entity", "size"),
    )
    agg["primary_share"] = (agg["n_primary_headlines"] /
                            agg["n_total"].where(agg["n_total"] > 0, 1.0))
    agg = agg.reindex(sessions, fill_value=0.0)
    agg["primary_share"] = agg["primary_share"].fillna(0.0)
    return agg[RELEVANCE_FEATURES]


# ---------------------------------------------------------------- combined

def build_features(
    bars: pd.DataFrame,
    headlines: pd.DataFrame,
    finbert: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Base feature table (price + aggregate-headline). Template-impact columns
    must be added separately with `template_impact.add_template_scores()` from
    inside the CV loop, since they need fold-aware OOF computation.
    """
    assert bars["bar_ix"].max() <= SEEN_MAX, (
        f"bars contain bar_ix > {SEEN_MAX}; feature builder must never see forward bars"
    )
    if finbert is None:
        finbert = load_finbert()
    price = build_price_features(bars)
    headline = build_headline_features(headlines, price.index.values, finbert=finbert)
    out = price.join(headline, how="left").fillna(0.0)
    base_cols = PRICE_FEATURES + HEADLINE_FEATURES
    assert list(out.columns) == base_cols, f"unexpected feature order: {list(out.columns)}"
    # Add template-score columns as zeros (placeholders); they get filled by
    # template_impact.add_template_scores when used inside CV / predict.
    for col in TEMPLATE_FEATURES:
        out[col] = 0.0
    return out


# ---------------------------------------------------------------- train target

def compute_train_target(
    bars_seen: pd.DataFrame, bars_unseen: pd.DataFrame
) -> pd.Series:
    """r_i = close_end / close_halfway - 1, per session, from bars_seen+bars_unseen."""
    seen_half = (
        bars_seen[bars_seen["bar_ix"] == HALFWAY_BAR]
        .set_index("session")["close"]
        .rename("c_half")
    )
    end = (
        bars_unseen[bars_unseen["bar_ix"] == END_BAR]
        .set_index("session")["close"]
        .rename("c_end")
    )
    df = pd.concat([seen_half, end], axis=1, join="inner")
    r = (df["c_end"] / df["c_half"]) - 1.0
    r.name = "r"
    return r
