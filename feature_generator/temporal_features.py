"""Temporal integration features: candle x headline interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-8


def generate_temporal_interaction_features(
    candles: pd.DataFrame,
    headlines: pd.DataFrame,
) -> dict[str, float]:
    """Generate ~10 interaction features connecting candle returns to headlines."""
    feats: dict[str, float] = {}
    close = candles["close"].values
    n_candles = len(close)
    n_rets = n_candles - 1
    rets = np.diff(close) / (np.abs(close[:-1]) + EPS)  # length n_candles-1

    if headlines.empty or n_rets == 0:
        feats["return_after_headline_mean"] = 0.0
        feats["return_after_headline_std"] = 0.0
        feats["return_before_headline_mean"] = 0.0
        feats["high_amplitude_return_mean"] = 0.0
        feats["vol_during_headline_candles"] = 0.0
        feats["vol_during_non_headline_candles"] = 0.0
        feats["headline_price_corr"] = 0.0
        feats["max_return_post_headline"] = 0.0
        feats["min_return_post_headline"] = 0.0
        feats["headline_cluster_count"] = 0.0
        return feats

    cidx = headlines["candle_idx"].values
    amp = headlines["amplitude"].values

    # Returns after headlines (1-candle forward)
    after_idx = cidx[cidx + 1 < n_candles]
    if len(after_idx) > 0:
        after_rets = rets[after_idx]  # rets[i] = return from candle i to candle i+1
        feats["return_after_headline_mean"] = float(np.mean(after_rets))
        feats["return_after_headline_std"] = float(np.std(after_rets, ddof=1)) if len(after_rets) > 1 else 0.0
    else:
        feats["return_after_headline_mean"] = 0.0
        feats["return_after_headline_std"] = 0.0

    # Returns before headlines (1-candle lookback)
    before_idx = cidx[cidx >= 1]
    if len(before_idx) > 0:
        before_rets = rets[before_idx - 1]
        feats["return_before_headline_mean"] = float(np.mean(before_rets))
    else:
        feats["return_before_headline_mean"] = 0.0

    # High amplitude return mean (amplitude > 0.7)
    high_amp_mask = amp > 0.7
    high_amp_cidx = cidx[high_amp_mask]
    high_amp_cidx = high_amp_cidx[high_amp_cidx + 1 < n_candles]
    if len(high_amp_cidx) > 0:
        feats["high_amplitude_return_mean"] = float(np.mean(rets[high_amp_cidx]))
    else:
        feats["high_amplitude_return_mean"] = 0.0

    # Volatility during headline vs non-headline candles
    hl_candle_set = set(cidx)
    hl_ret_mask = np.array([(i + 1) in hl_candle_set or i in hl_candle_set for i in range(n_rets)])

    if np.any(hl_ret_mask) and np.sum(hl_ret_mask) > 1:
        feats["vol_during_headline_candles"] = float(np.std(rets[hl_ret_mask], ddof=1))
    else:
        feats["vol_during_headline_candles"] = 0.0

    non_hl_ret_mask = ~hl_ret_mask
    if np.any(non_hl_ret_mask) and np.sum(non_hl_ret_mask) > 1:
        feats["vol_during_non_headline_candles"] = float(np.std(rets[non_hl_ret_mask], ddof=1))
    else:
        feats["vol_during_non_headline_candles"] = 0.0

    # Correlation: headline amplitude vs candle return
    valid = cidx < n_rets
    if np.sum(valid) >= 2:
        paired_rets = rets[cidx[valid]]
        paired_amps = amp[valid]
        # Guard against zero-variance vectors (e.g., amplitude fixed to 1.0),
        # which otherwise trigger runtime warnings and NaN correlations.
        if np.std(paired_amps) < 1e-12 or np.std(paired_rets) < 1e-12:
            feats["headline_price_corr"] = 0.0
        else:
            corr_matrix = np.corrcoef(paired_amps, paired_rets)
            corr_val = corr_matrix[0, 1]
            feats["headline_price_corr"] = float(corr_val) if np.isfinite(corr_val) else 0.0
    else:
        feats["headline_price_corr"] = 0.0

    # Max/min return in 3 candles after any headline
    post_indices = set()
    for ci in cidx:
        for offset in range(1, 4):
            idx = ci + offset
            if 0 <= idx < n_rets:
                post_indices.add(idx)
    if post_indices:
        post_rets = rets[sorted(post_indices)]
        feats["max_return_post_headline"] = float(np.max(post_rets))
        feats["min_return_post_headline"] = float(np.min(post_rets))
    else:
        feats["max_return_post_headline"] = 0.0
        feats["min_return_post_headline"] = 0.0

    # Headline cluster count: bursts of 2+ headlines within 3 consecutive candles
    feats["headline_cluster_count"] = float(_count_clusters(cidx))

    return feats


def _count_clusters(candle_indices: np.ndarray) -> int:
    """Count 'bursts': groups of 2+ headlines within 3 consecutive candles."""
    if len(candle_indices) < 2:
        return 0

    unique_candles = np.sort(np.unique(candle_indices))
    if len(unique_candles) < 2:
        return 1 if len(candle_indices) >= 2 else 0

    clusters = 0
    cluster_start = unique_candles[0]
    cluster_count = 1

    for i in range(1, len(unique_candles)):
        if unique_candles[i] - unique_candles[i - 1] <= 3:
            cluster_count += 1
        else:
            if cluster_count >= 2 or _headlines_in_range(candle_indices, cluster_start, unique_candles[i - 1]) >= 2:
                clusters += 1
            cluster_start = unique_candles[i]
            cluster_count = 1

    if cluster_count >= 2 or _headlines_in_range(candle_indices, cluster_start, unique_candles[-1]) >= 2:
        clusters += 1

    return clusters


def _headlines_in_range(candle_indices: np.ndarray, start: int, end: int) -> int:
    return int(np.sum((candle_indices >= start) & (candle_indices <= end)))
