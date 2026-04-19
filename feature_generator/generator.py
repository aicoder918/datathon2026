"""Main entry point for the feature generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_generator.candle_features import generate_candle_features
from feature_generator.headline_features import generate_headline_features
from feature_generator.temporal_features import generate_temporal_interaction_features


def generate_features(
    candles: pd.DataFrame,
    headlines: pd.DataFrame,
    n_templates: int = 30,
    recency_decay: float = 0.05,
) -> pd.Series:
    """Generate ~200 features for one session.

    Parameters
    ----------
    candles : DataFrame with columns [open, high, low, close].
        Can be any length (10–50+). Features for windows larger than
        len(candles) will be NaN (handled natively by LightGBM/CatBoost).
    headlines : DataFrame with columns [candle_idx, template_id, amplitude].
    n_templates : number of distinct headline templates.
    recency_decay : exponential decay rate for headline recency weighting.

    Returns
    -------
    pd.Series with ~200 named float features.  Values may include NaN for
    windowed features when the input history is shorter than the window.
    """
    n_candles = len(candles)
    feats: dict[str, float] = {}
    feats.update(generate_candle_features(candles))
    feats.update(generate_headline_features(headlines, n_templates, recency_decay, n_candles))
    feats.update(generate_temporal_interaction_features(candles, headlines))

    result = pd.Series(feats, dtype=float)
    # Replace inf with NaN; leave NaN as-is for tree models to handle natively
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
