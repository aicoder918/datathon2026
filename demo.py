"""Demo script: generate features from synthetic data and print summary."""

import numpy as np
import pandas as pd

from feature_generator import generate_features

N_TEMPLATES = 30
RECENCY_DECAY = 0.05


def make_synthetic_candles(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    opn = close + rng.normal(0, 0.1, n)
    high = np.maximum(close, opn) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(close, opn) - np.abs(rng.normal(0, 0.3, n))
    volume = rng.integers(1000, 10000, n).astype(float)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_synthetic_headlines(n_headlines: int = 25, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "candle_idx": rng.integers(0, 50, n_headlines),
        "template_id": rng.integers(0, N_TEMPLATES, n_headlines),
        "amplitude": rng.uniform(0, 1, n_headlines),
    })


if __name__ == "__main__":
    candles = make_synthetic_candles()
    headlines = make_synthetic_headlines()

    features = generate_features(
        candles=candles,
        headlines=headlines,
        n_templates=N_TEMPLATES,
        recency_decay=RECENCY_DECAY,
    )

    print(f"Total features: {len(features)}")
    print(f"All finite: {np.all(np.isfinite(features.values))}")
    print(f"dtype: {features.dtype}")
    print()
    print("Sample features (first 20):")
    print(features.head(20).to_string())
    print()
    print("Sample features (last 20):")
    print(features.tail(20).to_string())
