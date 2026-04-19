"""Candle-based features: price-level, returns, trend, volatility, technicals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WINDOWS = [5, 10, 20, 50]
EPS = 1e-8

_NAN = float("nan")


def generate_candle_features(candles: pd.DataFrame) -> dict[str, float]:
    """Generate ~90 features from an OHLCV DataFrame of any length."""
    feats: dict[str, float] = {}
    feats.update(_price_level_features(candles))
    feats.update(_returns_features(candles))
    feats.update(_trend_features(candles))
    feats.update(_volatility_features(candles))
    feats.update(_technical_indicators(candles))
    return feats


# ---------------------------------------------------------------------------
# 1.1  Price-level features  (~32)
# ---------------------------------------------------------------------------
def _price_level_features(candles: pd.DataFrame) -> dict[str, float]:
    close = candles["close"].values
    high = candles["high"].values
    low = candles["low"].values
    opn = candles["open"].values
    n_candles = len(close)
    feats: dict[str, float] = {}
    for n in WINDOWS:
        if n_candles < n:
            for name in (f"close_mean_w{n}", f"close_std_w{n}", f"close_min_w{n}",
                         f"close_max_w{n}", f"close_skew_w{n}", f"close_kurt_w{n}",
                         f"high_low_range_mean_w{n}", f"close_vs_open_w{n}"):
                feats[name] = _NAN
            continue
        c = close[-n:]
        h = high[-n:]
        lo = low[-n:]
        o = opn[-n:]
        feats[f"close_mean_w{n}"] = float(np.mean(c))
        feats[f"close_std_w{n}"] = float(np.std(c, ddof=1)) if n > 1 else 0.0
        feats[f"close_min_w{n}"] = float(np.min(c))
        feats[f"close_max_w{n}"] = float(np.max(c))
        feats[f"close_skew_w{n}"] = float(stats.skew(c, bias=False)) if n > 2 else 0.0
        feats[f"close_kurt_w{n}"] = float(stats.kurtosis(c, bias=False)) if n > 3 else 0.0
        feats[f"high_low_range_mean_w{n}"] = float(np.mean(h - lo))
        co = (c - o) / (np.abs(o) + EPS)
        feats[f"close_vs_open_w{n}"] = float(np.mean(co))
    return feats


# ---------------------------------------------------------------------------
# 1.2  Returns features  (~20)
# ---------------------------------------------------------------------------
def _returns_features(candles: pd.DataFrame) -> dict[str, float]:
    close = candles["close"].values
    n_candles = len(close)
    rets = np.diff(close) / (np.abs(close[:-1]) + EPS)
    feats: dict[str, float] = {}
    for n in WINDOWS:
        if n_candles < n:
            for name in (f"return_mean_w{n}", f"return_std_w{n}", f"return_skew_w{n}",
                         f"return_positive_ratio_w{n}", f"return_cumulative_w{n}"):
                feats[name] = _NAN
            continue
        r = rets[-(n - 1):] if n > 1 else rets[-1:]
        if len(r) == 0:
            r = np.array([0.0])
        feats[f"return_mean_w{n}"] = float(np.mean(r))
        feats[f"return_std_w{n}"] = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        feats[f"return_skew_w{n}"] = float(stats.skew(r, bias=False)) if len(r) > 2 else 0.0
        feats[f"return_positive_ratio_w{n}"] = float(np.mean(r > 0))
        feats[f"return_cumulative_w{n}"] = float(np.prod(1 + r) - 1)
    return feats


# ---------------------------------------------------------------------------
# 1.3  Trend features  (~12)
# ---------------------------------------------------------------------------
def _trend_features(candles: pd.DataFrame) -> dict[str, float]:
    close = candles["close"].values
    n_candles = len(close)
    feats: dict[str, float] = {}
    for n in WINDOWS:
        if n_candles < n:
            for name in (f"slope_linreg_w{n}", f"slope_pvalue_w{n}", f"trend_consistency_w{n}"):
                feats[name] = _NAN
            continue
        c = close[-n:]
        x = np.arange(n, dtype=float)
        if n >= 2:
            slope, _intercept, _r, pvalue, _stderr = stats.linregress(x, c)
            feats[f"slope_linreg_w{n}"] = float(slope)
            feats[f"slope_pvalue_w{n}"] = float(pvalue)
        else:
            feats[f"slope_linreg_w{n}"] = 0.0
            feats[f"slope_pvalue_w{n}"] = 1.0
        # trend consistency: fraction of consecutive same-direction moves
        diffs = np.diff(c)
        if len(diffs) >= 2:
            same_dir = np.sum(diffs[1:] * diffs[:-1] > 0)
            feats[f"trend_consistency_w{n}"] = float(same_dir / (len(diffs) - 1))
        else:
            feats[f"trend_consistency_w{n}"] = 0.0
    return feats


# ---------------------------------------------------------------------------
# 1.4  Volatility features  (~12)
# ---------------------------------------------------------------------------
def _volatility_features(candles: pd.DataFrame) -> dict[str, float]:
    close = candles["close"].values
    high = candles["high"].values
    low = candles["low"].values
    opn = candles["open"].values
    n_candles = len(close)
    rets = np.diff(close) / (np.abs(close[:-1]) + EPS)
    feats: dict[str, float] = {}
    for n in WINDOWS:
        if n_candles < n:
            for name in (f"atr_w{n}", f"realized_vol_w{n}", f"garman_klass_vol_w{n}"):
                feats[name] = _NAN
            continue
        c = close[-n:]
        h = high[-n:]
        lo = low[-n:]
        o = opn[-n:]
        r = rets[-(n - 1):] if n > 1 else rets[-1:]
        # ATR: mean of max(H-L, |H-Cprev|, |L-Cprev|)
        c_prev = close[-(n):-1] if n > 1 else close[-1:]
        if len(c_prev) == len(h) - 1 and len(h) > 1:
            tr = np.maximum(
                h[1:] - lo[1:],
                np.maximum(np.abs(h[1:] - c_prev), np.abs(lo[1:] - c_prev)),
            )
            feats[f"atr_w{n}"] = float(np.mean(tr))
        else:
            feats[f"atr_w{n}"] = float(np.mean(h - lo))
        # realized vol
        feats[f"realized_vol_w{n}"] = float(np.std(r, ddof=1) * np.sqrt(n)) if len(r) > 1 else 0.0
        # Garman-Klass volatility
        log_hl = np.log((h + EPS) / (lo + EPS))
        log_co = np.log((c + EPS) / (o + EPS))
        gk = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
        feats[f"garman_klass_vol_w{n}"] = float(np.mean(gk))
    return feats


# ---------------------------------------------------------------------------
# 1.5  Technical indicators  (~14, computed at t50)
# ---------------------------------------------------------------------------
def _technical_indicators(candles: pd.DataFrame) -> dict[str, float]:
    close = pd.Series(candles["close"].values, dtype=float)
    high = pd.Series(candles["high"].values, dtype=float)
    low = pd.Series(candles["low"].values, dtype=float)
    feats: dict[str, float] = {}

    # RSI 14
    feats["rsi_14"] = _rsi(close, 14)

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feats["macd"] = float(macd_line.iloc[-1])
    feats["macd_signal"] = float(macd_signal.iloc[-1])
    feats["macd_histogram"] = float(macd_line.iloc[-1] - macd_signal.iloc[-1])

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=1)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    last_close = close.iloc[-1]
    feats["bollinger_upper_20"] = float(bb_upper.iloc[-1]) if not np.isnan(bb_upper.iloc[-1]) else float(last_close)
    feats["bollinger_lower_20"] = float(bb_lower.iloc[-1]) if not np.isnan(bb_lower.iloc[-1]) else float(last_close)
    feats["bollinger_width_20"] = float(bb_width.iloc[-1]) if not np.isnan(bb_width.iloc[-1]) else 0.0
    bb_range = feats["bollinger_upper_20"] - feats["bollinger_lower_20"]
    feats["close_bollinger_pct_20"] = float(
        (last_close - feats["bollinger_lower_20"]) / (bb_range + EPS)
    )

    # EMAs and SMAs
    for span in [5, 10, 20]:
        ema_val = close.ewm(span=span, adjust=False).mean().iloc[-1]
        sma_val = close.rolling(span).mean().iloc[-1]
        feats[f"ema_{span}"] = float(ema_val)
        feats[f"sma_{span}"] = float(sma_val) if not np.isnan(sma_val) else float(last_close)

    # close vs EMA20
    ema20_val = feats["ema_20"]
    feats["close_vs_ema20"] = float((last_close - ema20_val) / (abs(ema20_val) + EPS))

    return feats


def _rsi(close: pd.Series, period: int) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + EPS)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0
