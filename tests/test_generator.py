"""Tests for the feature_generator package."""

import numpy as np
import pandas as pd
import pytest

from feature_generator import generate_features
from feature_generator.candle_features import generate_candle_features
from feature_generator.headline_features import generate_headline_features
from feature_generator.temporal_features import generate_temporal_interaction_features

N_TEMPLATES = 30
RECENCY_DECAY = 0.05


def _make_candles(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create synthetic candles via a random walk."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    opn = close + rng.normal(0, 0.1, n)
    high = np.maximum(close, opn) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(close, opn) - np.abs(rng.normal(0, 0.3, n))
    volume = rng.integers(1000, 10000, n).astype(float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_headlines(n_headlines: int = 20, seed: int = 42) -> pd.DataFrame:
    """Create random headlines."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "candle_idx": rng.integers(0, 50, n_headlines),
            "template_id": rng.integers(0, N_TEMPLATES, n_headlines),
            "amplitude": rng.uniform(0, 1, n_headlines),
        }
    )


def _make_empty_headlines() -> pd.DataFrame:
    return pd.DataFrame(columns=["candle_idx", "template_id", "amplitude"])


class TestFullPipeline:
    def test_feature_count_in_range(self):
        candles = _make_candles()
        headlines = _make_headlines()
        feats = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        assert 190 <= len(feats) <= 210, f"Expected ~200 features, got {len(feats)}"

    def test_all_values_finite(self):
        candles = _make_candles()
        headlines = _make_headlines()
        feats = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        assert np.all(np.isfinite(feats.values)), "Found non-finite values"

    def test_all_values_are_float(self):
        candles = _make_candles()
        headlines = _make_headlines()
        feats = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        assert feats.dtype == np.float64

    def test_unique_feature_names(self):
        candles = _make_candles()
        headlines = _make_headlines()
        feats = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        assert len(feats.index) == len(set(feats.index)), "Duplicate feature names"

    def test_deterministic(self):
        candles = _make_candles()
        headlines = _make_headlines()
        f1 = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        f2 = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        pd.testing.assert_series_equal(f1, f2)


class TestEmptyHeadlines:
    def test_feature_count_with_empty_headlines(self):
        candles = _make_candles()
        headlines = _make_empty_headlines()
        feats = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        assert 190 <= len(feats) <= 210

    def test_all_values_finite_with_empty_headlines(self):
        candles = _make_candles()
        headlines = _make_empty_headlines()
        feats = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        assert np.all(np.isfinite(feats.values))

    def test_recency_defaults_to_50(self):
        candles = _make_candles()
        headlines = _make_empty_headlines()
        feats = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        for tid in range(N_TEMPLATES):
            assert feats[f"t{tid}_recency"] == 50.0

    def test_headline_counts_zero(self):
        candles = _make_candles()
        headlines = _make_empty_headlines()
        feats = generate_features(candles, headlines, N_TEMPLATES, RECENCY_DECAY)
        assert feats["total_headlines"] == 0.0
        assert feats["total_headlines_last10"] == 0.0


class TestCandleFeatures:
    def test_candle_feature_count(self):
        candles = _make_candles()
        feats = generate_candle_features(candles)
        # 32 price-level + 20 returns + 12 trend + 12 volatility + 14 tech = 90
        assert len(feats) >= 85, f"Expected ~90 candle features, got {len(feats)}"

    def test_window_features_present(self):
        candles = _make_candles()
        feats = generate_candle_features(candles)
        for n in [5, 10, 20, 50]:
            assert f"close_mean_w{n}" in feats
            assert f"return_mean_w{n}" in feats
            assert f"slope_linreg_w{n}" in feats
            assert f"atr_w{n}" in feats

    def test_technical_indicators_present(self):
        candles = _make_candles()
        feats = generate_candle_features(candles)
        for key in ["rsi_14", "macd", "macd_signal", "macd_histogram",
                     "bollinger_upper_20", "bollinger_lower_20",
                     "ema_5", "ema_10", "ema_20", "sma_5", "sma_10", "sma_20",
                     "close_vs_ema20"]:
            assert key in feats, f"Missing {key}"

    def test_rsi_in_range(self):
        candles = _make_candles()
        feats = generate_candle_features(candles)
        assert 0 <= feats["rsi_14"] <= 100


class TestHeadlineFeatures:
    def test_per_template_features(self):
        headlines = _make_headlines()
        feats = generate_headline_features(headlines, N_TEMPLATES, RECENCY_DECAY)
        for tid in range(N_TEMPLATES):
            assert f"t{tid}_count_total" in feats
            assert f"t{tid}_amplitude_sum_weighted" in feats
            assert f"t{tid}_recency" in feats

    def test_global_features_present(self):
        headlines = _make_headlines()
        feats = generate_headline_features(headlines, N_TEMPLATES, RECENCY_DECAY)
        for key in ["total_headlines", "total_headlines_last10",
                     "unique_templates_count", "unique_templates_last10",
                     "mean_amplitude_all", "max_amplitude_all",
                     "headline_density_first_half", "headline_density_second_half",
                     "amplitude_trend", "candles_since_last_headline"]:
            assert key in feats, f"Missing {key}"

    def test_total_headlines_correct(self):
        headlines = _make_headlines(n_headlines=15)
        feats = generate_headline_features(headlines, N_TEMPLATES, RECENCY_DECAY)
        assert feats["total_headlines"] == 15.0


class TestTemporalFeatures:
    def test_temporal_feature_count(self):
        candles = _make_candles()
        headlines = _make_headlines()
        feats = generate_temporal_interaction_features(candles, headlines)
        assert len(feats) == 10

    def test_temporal_features_present(self):
        candles = _make_candles()
        headlines = _make_headlines()
        feats = generate_temporal_interaction_features(candles, headlines)
        for key in ["return_after_headline_mean", "return_after_headline_std",
                     "return_before_headline_mean", "high_amplitude_return_mean",
                     "vol_during_headline_candles", "vol_during_non_headline_candles",
                     "headline_price_corr", "max_return_post_headline",
                     "min_return_post_headline", "headline_cluster_count"]:
            assert key in feats, f"Missing {key}"

    def test_empty_headlines_temporal(self):
        candles = _make_candles()
        headlines = _make_empty_headlines()
        feats = generate_temporal_interaction_features(candles, headlines)
        assert all(np.isfinite(v) for v in feats.values())
