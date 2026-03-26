"""Technical indicator computation helpers.

Thin wrappers around pandas-ta-classic and manual implementations for
indicators used across all strategies.
"""

from __future__ import annotations

import pandas as pd


def compute_atr(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Compute Average True Range (ATR) using Wilder's smoothing.

    Args:
        highs:  Series of high prices.
        lows:   Series of low prices.
        closes: Series of close prices.
        length: ATR period (default 14).

    Returns:
        pd.Series of ATR values (NaN for first ``length-1`` rows).
    """
    high = highs.reset_index(drop=True)
    low = lows.reset_index(drop=True)
    close = closes.reset_index(drop=True)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing (RMA)
    atr = tr.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    return atr


def compute_ema(closes: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average.

    Args:
        closes: Series of close prices.
        period: EMA period.

    Returns:
        pd.Series of EMA values.
    """
    return closes.ewm(span=period, adjust=False).mean()


def compute_sma(closes: pd.Series, period: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return closes.rolling(window=period).mean()


def compute_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index.

    Args:
        closes: Series of close prices.
        period: RSI period (default 14).

    Returns:
        pd.Series of RSI values in [0, 100].
    """
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_vwap(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    volumes: pd.Series,
) -> pd.Series:
    """Compute Volume Weighted Average Price (cumulative).

    Args:
        highs:   Series of high prices.
        lows:    Series of low prices.
        closes:  Series of close prices.
        volumes: Series of volume values.

    Returns:
        pd.Series of VWAP values.
    """
    typical_price = (highs + lows + closes) / 3.0
    cum_vol = volumes.cumsum()
    cum_tpv = (typical_price * volumes).cumsum()
    return cum_tpv / cum_vol.replace(0, float("nan"))


def compute_bollinger_bands(
    closes: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    middle = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def candle_body_size(opens: pd.Series, closes: pd.Series) -> pd.Series:
    """Return absolute candle body size."""
    return (closes - opens).abs()


def is_bullish(opens: pd.Series, closes: pd.Series) -> pd.Series:
    """Return boolean Series: True where close > open."""
    return closes > opens
