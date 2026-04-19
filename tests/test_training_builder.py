"""Tests for the training_builder augmentation module."""

import numpy as np
import pandas as pd
import pytest

from feature_generator import generate_features
from feature_generator.training_builder import (
    build_training_rows,
    build_training_dataset,
    compute_sample_weights,
)

N_TEMPLATES = 30
RECENCY_DECAY = 0.05


def _make_candles(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    opn = close + rng.normal(0, 0.1, n)
    high = np.maximum(close, opn) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(close, opn) - np.abs(rng.normal(0, 0.3, n))
    return pd.DataFrame({
        "open": opn, "high": high, "low": low, "close": close,
    })


def _make_headlines(n_headlines: int = 25, max_candle: int = 49, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "candle_idx": rng.integers(0, max_candle + 1, n_headlines),
        "template_id": rng.integers(0, N_TEMPLATES, n_headlines),
        "amplitude": rng.uniform(0, 1, n_headlines),
    })


def _make_empty_headlines() -> pd.DataFrame:
    return pd.DataFrame(columns=["candle_idx", "template_id", "amplitude"])


# ------------------------------------------------------------------ #
# build_training_rows
# ------------------------------------------------------------------ #

class TestBuildTrainingRows:
    def test_produces_5_rows_default(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        assert len(rows) == 5

    def test_history_length_present(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        for row in rows:
            assert "history_length" in row.index

    def test_history_length_values(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        expected = [10, 20, 30, 40, 50]
        actual = [int(r["history_length"]) for r in rows]
        assert actual == expected

    def test_target_present(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        for row in rows:
            assert "target" in row.index
            assert np.isfinite(row["target"])

    def test_target_values_correct(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES,
                                   horizon=50)
        # Row with h=10: target = candles.iloc[10+50-1=59]["close"]
        assert rows[0]["target"] == pytest.approx(candles.iloc[59]["close"])
        # Row with h=50: target = candles.iloc[50+50-1=99]["close"]
        assert rows[4]["target"] == pytest.approx(candles.iloc[99]["close"])

    def test_no_target_when_disabled(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES,
                                   include_target=False)
        for row in rows:
            assert "target" not in row.index

    def test_session_id_preserved(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES,
                                   session_id=42)
        for row in rows:
            assert row["session_id"] == 42

    def test_raises_on_insufficient_candles(self):
        candles = _make_candles(55)  # h=10 + horizon=50 = 60 > 55
        headlines = _make_headlines()
        with pytest.raises(ValueError, match="Cannot compute target"):
            build_training_rows(candles, headlines, n_templates=N_TEMPLATES,
                                history_lengths=(10,), horizon=50)

    def test_empty_headlines(self):
        candles = _make_candles(100)
        headlines = _make_empty_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        assert len(rows) == 5
        for row in rows:
            assert np.isfinite(row["target"])

    def test_feature_names_consistent_across_rows(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        # All rows should have the same feature names (superset)
        names_set = rows[0].index
        for row in rows[1:]:
            assert set(row.index) == set(names_set)


# ------------------------------------------------------------------ #
# NaN for oversized windows on short history
# ------------------------------------------------------------------ #

class TestShortHistoryNaN:
    def test_w50_nan_on_10_candle_history(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        # h=10 row: window 50 features should be NaN
        row_h10 = rows[0]
        assert np.isnan(row_h10["close_mean_w50"])
        assert np.isnan(row_h10["return_mean_w50"])
        assert np.isnan(row_h10["slope_linreg_w50"])
        assert np.isnan(row_h10["atr_w50"])

    def test_w20_nan_on_10_candle_history(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        row_h10 = rows[0]
        assert np.isnan(row_h10["close_mean_w20"])

    def test_w5_valid_on_10_candle_history(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        row_h10 = rows[0]
        assert np.isfinite(row_h10["close_mean_w5"])
        assert np.isfinite(row_h10["return_mean_w5"])

    def test_all_windows_valid_on_full_history(self):
        candles = _make_candles(100)
        headlines = _make_headlines()
        rows = build_training_rows(candles, headlines, n_templates=N_TEMPLATES)
        row_h50 = rows[4]  # h=50
        assert np.isfinite(row_h50["close_mean_w50"])
        assert np.isfinite(row_h50["close_mean_w5"])


# ------------------------------------------------------------------ #
# build_training_dataset
# ------------------------------------------------------------------ #

class TestBuildTrainingDataset:
    def test_concatenates_cleanly(self):
        sessions = [
            (_make_candles(100, seed=i), _make_headlines(seed=i))
            for i in range(3)
        ]
        df = build_training_dataset(sessions, n_templates=N_TEMPLATES)
        # 3 sessions * 5 history lengths = 15 rows
        assert len(df) == 15

    def test_session_id_column(self):
        sessions = [
            (_make_candles(100, seed=i), _make_headlines(seed=i))
            for i in range(3)
        ]
        df = build_training_dataset(sessions, session_ids=[100, 200, 300],
                                    n_templates=N_TEMPLATES)
        assert "session_id" in df.columns
        assert set(df["session_id"]) == {100, 200, 300}
        # Each session has 5 rows
        assert (df.groupby("session_id").size() == 5).all()

    def test_has_target_and_history_length(self):
        sessions = [(_make_candles(100), _make_headlines())]
        df = build_training_dataset(sessions, n_templates=N_TEMPLATES)
        assert "target" in df.columns
        assert "history_length" in df.columns

    def test_all_columns_consistent(self):
        sessions = [
            (_make_candles(100, seed=i), _make_headlines(seed=i))
            for i in range(5)
        ]
        df = build_training_dataset(sessions, n_templates=N_TEMPLATES)
        # No unexpected all-NaN columns that shouldn't exist
        assert df.shape[0] == 25
        assert df.shape[1] > 200  # features + history_length + target + session_id


# ------------------------------------------------------------------ #
# compute_sample_weights
# ------------------------------------------------------------------ #

class TestSampleWeights:
    def test_uniform(self):
        hl = [10, 10, 20, 20, 50, 50]
        w = compute_sample_weights(hl, strategy="uniform")
        np.testing.assert_array_equal(w, np.ones(6))

    def test_balanced_equal_counts(self):
        hl = [10, 20, 50] * 4  # 4 each
        w = compute_sample_weights(hl, strategy="balanced")
        # All weights should be equal since counts are equal
        assert np.allclose(w, w[0])

    def test_balanced_unequal_counts(self):
        hl = [10, 10, 10, 50]  # 3x10, 1x50
        w = compute_sample_weights(hl, strategy="balanced")
        # Total weight per group should be equal
        w_10 = w[:3].sum()
        w_50 = w[3:].sum()
        assert pytest.approx(w_10) == w_50

    def test_favor_full(self):
        hl = [10, 20, 50]
        w = compute_sample_weights(hl, strategy="favor_full")
        # w50 should be 2x the balanced weight
        w_balanced = compute_sample_weights(hl, strategy="balanced")
        assert w[2] == pytest.approx(w_balanced[2] * 2)
        assert w[0] == pytest.approx(w_balanced[0])

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_sample_weights([10, 20], strategy="invalid")


# ------------------------------------------------------------------ #
# Backward compatibility: 50-candle generate_features still works
# ------------------------------------------------------------------ #

class TestBackwardCompatibility:
    def test_50_candle_no_nan(self):
        """The original 50-candle path should produce no NaN in non-window features."""
        candles = _make_candles(50)
        headlines = _make_headlines()
        feats = generate_features(candles, headlines, n_templates=N_TEMPLATES)
        # All w5, w10, w20, w50 should be finite since we have 50 candles
        for n in [5, 10, 20, 50]:
            assert np.isfinite(feats[f"close_mean_w{n}"])
            assert np.isfinite(feats[f"return_mean_w{n}"])
