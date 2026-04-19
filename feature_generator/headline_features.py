"""Headline / template features: per-template and global aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8


def generate_headline_features(
    headlines: pd.DataFrame,
    n_templates: int = 30,
    recency_decay: float = 0.05,
    n_candles: int = 50,
) -> dict[str, float]:
    """Generate ~100 features from the headlines DataFrame.

    Parameters
    ----------
    headlines : DataFrame with columns [candle_idx, template_id, amplitude]
        and optional `sentiment` (template-level signed score).
    n_templates : number of distinct templates (0 .. n_templates-1).
    recency_decay : kept for API compatibility; no longer used here.
    n_candles : number of candles in the input window (used for recency
        calculations instead of hardcoded 49).
    """
    feats: dict[str, float] = {}
    feats.update(_per_template_features(headlines, n_templates, recency_decay, n_candles))
    feats.update(_global_headline_features(headlines, n_candles))
    return feats


# ---------------------------------------------------------------------------
# 2.1  Per-template features  (3 × n_templates)
# ---------------------------------------------------------------------------
def _per_template_features(
    headlines: pd.DataFrame,
    n_templates: int,
    recency_decay: float,
    n_candles: int,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    is_empty = headlines.empty
    last_idx = n_candles - 1  # index of the last candle

    for tid in range(n_templates):
        prefix = f"t{tid}"
        if is_empty:
            feats[f"{prefix}_count_total"] = 0.0
            feats[f"{prefix}_sentiment_mean"] = 0.0
            feats[f"{prefix}_recency"] = float(n_candles)
            continue

        mask = headlines["template_id"] == tid
        subset = headlines.loc[mask]

        # count
        feats[f"{prefix}_count_total"] = float(len(subset))

        if len(subset) == 0:
            feats[f"{prefix}_sentiment_mean"] = 0.0
            feats[f"{prefix}_recency"] = float(n_candles)
            continue

        # Average template impact/sentiment within the current session.
        # This intentionally ignores dollar/percentage-derived amplitudes.
        candle_idx = subset["candle_idx"].values
        if "sentiment" in subset.columns:
            sent = pd.to_numeric(subset["sentiment"], errors="coerce").fillna(0.0).values
        else:
            sent = np.zeros(len(subset), dtype=float)
        feats[f"{prefix}_sentiment_mean"] = float(np.mean(sent))

        # recency: candles since last appearance
        feats[f"{prefix}_recency"] = float(last_idx - candle_idx.max())

    return feats


# ---------------------------------------------------------------------------
# 2.2  Global headline features  (~10)
# ---------------------------------------------------------------------------
def _global_headline_features(
    headlines: pd.DataFrame,
    n_candles: int,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    last_idx = n_candles - 1
    half = n_candles // 2
    last10_start = max(n_candles - 10, 0)

    if headlines.empty:
        feats["total_headlines"] = 0.0
        feats["total_headlines_last10"] = 0.0
        feats["unique_templates_count"] = 0.0
        feats["unique_templates_last10"] = 0.0
        feats["mean_amplitude_all"] = 0.0
        feats["max_amplitude_all"] = 0.0
        feats["hl_net_sent"] = 0.0
        feats["hl_net_sent_recent"] = 0.0
        feats["hl_mean_sent"] = 0.0
        feats["hl_n_pos"] = 0.0
        feats["hl_n_neg"] = 0.0
        feats["headline_density_first_half"] = 0.0
        feats["headline_density_second_half"] = 0.0
        feats["amplitude_trend"] = 0.0
        feats["candles_since_last_headline"] = float(n_candles)
        return feats

    cidx = headlines["candle_idx"].values
    amp = headlines["amplitude"].values
    tid = headlines["template_id"].values
    if "sentiment" in headlines.columns:
        sent = pd.to_numeric(headlines["sentiment"], errors="coerce").fillna(0.0).values
    else:
        sent = np.zeros(len(headlines), dtype=float)

    feats["total_headlines"] = float(len(headlines))

    last10_mask = cidx >= last10_start
    feats["total_headlines_last10"] = float(np.sum(last10_mask))

    feats["unique_templates_count"] = float(len(np.unique(tid)))
    if np.any(last10_mask):
        feats["unique_templates_last10"] = float(len(np.unique(tid[last10_mask])))
    else:
        feats["unique_templates_last10"] = 0.0

    feats["mean_amplitude_all"] = float(np.mean(amp))
    feats["max_amplitude_all"] = float(np.max(amp))
    feats["hl_net_sent"] = float(np.sum(sent))
    feats["hl_net_sent_recent"] = float(np.sum(sent[last10_mask])) if np.any(last10_mask) else 0.0
    feats["hl_mean_sent"] = float(np.mean(sent))
    feats["hl_n_pos"] = float(np.sum(sent > 0))
    feats["hl_n_neg"] = float(np.sum(sent < 0))

    feats["headline_density_first_half"] = float(np.sum(cidx < half))
    feats["headline_density_second_half"] = float(np.sum(cidx >= half))

    # amplitude trend: slope of amplitude vs candle_idx
    if len(headlines) >= 2 and len(np.unique(cidx)) >= 2:
        from scipy import stats as sp_stats

        slope, *_ = sp_stats.linregress(cidx.astype(float), amp.astype(float))
        feats["amplitude_trend"] = float(slope)
    else:
        feats["amplitude_trend"] = 0.0

    feats["candles_since_last_headline"] = float(last_idx - cidx.max())

    return feats
