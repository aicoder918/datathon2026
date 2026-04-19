"""Training data augmentation: multiple rows per session at different history lengths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_generator.generator import generate_features


def build_training_rows(
    candles: pd.DataFrame,
    headlines: pd.DataFrame,
    history_lengths: tuple[int, ...] = (10, 20, 30, 40, 50),
    horizon: int = 50,
    n_templates: int = 30,
    recency_decay: float = 0.05,
    include_history_length_feature: bool = True,
    include_target: bool = True,
    session_id: int | str | None = None,
) -> list[pd.Series]:
    """Generate one feature row per history length for a single session.

    Parameters
    ----------
    candles : DataFrame with >=max(history_lengths)+horizon rows, columns
        include at least [open, high, low, close].
    headlines : DataFrame with columns [candle_idx, template_id, amplitude].
    history_lengths : tuple of input window sizes to generate.
    horizon : how many candles ahead to predict from end of input window.
    n_templates : passed through to generate_features.
    recency_decay : passed through to generate_features.
    include_history_length_feature : if True, add ``history_length`` to each row.
    include_target : if True, compute and include the target close value.
    session_id : optional identifier, added to each row if provided.

    Returns
    -------
    List of pd.Series, one per history length.
    """
    total_candles = len(candles)
    rows: list[pd.Series] = []

    for h in history_lengths:
        target_idx = h + horizon - 1
        if include_target and target_idx >= total_candles:
            raise ValueError(
                f"Cannot compute target: history_length={h} + horizon={horizon} "
                f"requires candle index {target_idx}, but only {total_candles} "
                f"candles available."
            )

        # Slice candles to first h bars
        candles_slice = candles.iloc[:h].reset_index(drop=True)

        # Slice headlines to those within the input window
        if headlines.empty:
            hl_slice = headlines
        else:
            hl_slice = headlines.loc[headlines["candle_idx"] < h].copy()

        feats = generate_features(candles_slice, hl_slice, n_templates, recency_decay)

        if include_history_length_feature:
            feats["history_length"] = float(h)

        if include_target:
            feats["target"] = float(candles.iloc[target_idx]["close"])

        if session_id is not None:
            feats["session_id"] = session_id

        rows.append(feats)

    return rows


def build_training_dataset(
    sessions: list[tuple[pd.DataFrame, pd.DataFrame]],
    session_ids: list | None = None,
    history_lengths: tuple[int, ...] = (10, 20, 30, 40, 50),
    horizon: int = 50,
    n_templates: int = 30,
    recency_decay: float = 0.05,
) -> pd.DataFrame:
    """Build a full training DataFrame from multiple sessions.

    Parameters
    ----------
    sessions : list of (candles_df, headlines_df) tuples.
    session_ids : optional list of session identifiers (same length as sessions).
    history_lengths : tuple of input window sizes.
    horizon : prediction horizon in candles.
    n_templates : number of headline templates.
    recency_decay : headline recency decay rate.

    Returns
    -------
    DataFrame with ~200 feature columns + ``history_length``, ``target``,
    and ``session_id`` columns. Use ``session_id`` with GroupKFold for
    correct cross-validation.
    """
    if session_ids is not None and len(session_ids) != len(sessions):
        raise ValueError(
            f"session_ids length ({len(session_ids)}) != sessions length ({len(sessions)})"
        )

    all_rows: list[pd.Series] = []
    for i, (candles, headlines) in enumerate(sessions):
        sid = session_ids[i] if session_ids is not None else i
        rows = build_training_rows(
            candles=candles,
            headlines=headlines,
            history_lengths=history_lengths,
            horizon=horizon,
            n_templates=n_templates,
            recency_decay=recency_decay,
            include_history_length_feature=True,
            include_target=True,
            session_id=sid,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    return df


def compute_sample_weights(
    history_lengths: list[int],
    strategy: str = "balanced",
) -> np.ndarray:
    """Compute per-row weights to rebalance across history lengths.

    Parameters
    ----------
    history_lengths : list of history_length values, one per training row.
    strategy : one of 'balanced', 'favor_full', 'uniform'.
        - 'balanced': each history length group gets equal total weight.
        - 'favor_full': max(history_lengths) rows get 2x weight, rest balanced.
        - 'uniform': all rows weight 1.0.

    Returns
    -------
    np.ndarray of per-row weights.
    """
    hl = np.asarray(history_lengths, dtype=float)
    n = len(hl)

    if strategy == "uniform":
        return np.ones(n, dtype=float)

    unique_vals, counts = np.unique(hl, return_counts=True)
    n_groups = len(unique_vals)
    weight_per_group = {v: n / (n_groups * c) for v, c in zip(unique_vals, counts)}

    weights = np.array([weight_per_group[v] for v in hl], dtype=float)

    if strategy == "balanced":
        return weights

    if strategy == "favor_full":
        max_h = unique_vals.max()
        weights[hl == max_h] *= 2.0
        return weights

    raise ValueError(f"Unknown strategy: {strategy!r}. Use 'balanced', 'favor_full', or 'uniform'.")
