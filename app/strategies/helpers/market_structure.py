"""Market structure analysis helpers.

Higher-timeframe trend detection and structure break identification used
across multiple strategies.
"""

from __future__ import annotations

import pandas as pd

from app.strategies.helpers.indicators import compute_ema


def detect_trend(
    closes: pd.Series,
    fast_period: int = 50,
    slow_period: int = 200,
) -> str:
    """Detect trend direction using EMA crossover.

    Args:
        closes:      Series of close prices.
        fast_period: Fast EMA period (default 50).
        slow_period: Slow EMA period (default 200).

    Returns:
        ``"bullish"``, ``"bearish"``, or ``"neutral"``.
    """
    if len(closes) < slow_period:
        return "neutral"

    fast_ema = compute_ema(closes, fast_period)
    slow_ema = compute_ema(closes, slow_period)

    last_fast = float(fast_ema.iloc[-1])
    last_slow = float(slow_ema.iloc[-1])

    if last_fast > last_slow:
        return "bullish"
    elif last_fast < last_slow:
        return "bearish"
    return "neutral"


def detect_higher_highs_higher_lows(
    highs: pd.Series,
    lows: pd.Series,
    lookback: int = 20,
) -> str:
    """Detect HH/HL (uptrend) or LH/LL (downtrend) pattern.

    Args:
        highs:    Series of high prices.
        lows:     Series of low prices.
        lookback: Number of recent bars to evaluate.

    Returns:
        ``"uptrend"``, ``"downtrend"``, or ``"sideways"``.
    """
    if len(highs) < lookback:
        return "sideways"

    recent_highs = highs.iloc[-lookback:]
    recent_lows = lows.iloc[-lookback:]

    mid = lookback // 2

    first_half_high = float(recent_highs.iloc[:mid].max())
    second_half_high = float(recent_highs.iloc[mid:].max())
    first_half_low = float(recent_lows.iloc[:mid].min())
    second_half_low = float(recent_lows.iloc[mid:].min())

    if second_half_high > first_half_high and second_half_low > first_half_low:
        return "uptrend"
    elif second_half_high < first_half_high and second_half_low < first_half_low:
        return "downtrend"
    return "sideways"


def is_consolidating(
    highs: pd.Series,
    lows: pd.Series,
    atr: pd.Series,
    lookback: int = 10,
    atr_threshold: float = 0.5,
) -> bool:
    """Detect if price is in a consolidation zone.

    Consolidation is defined as the range of recent bars being less than
    ``atr_threshold`` * current ATR * lookback.

    Args:
        highs:         Series of high prices.
        lows:          Series of low prices.
        atr:           Series of ATR values.
        lookback:      Number of recent bars to evaluate.
        atr_threshold: Range relative to ATR (default 0.5).

    Returns:
        True if consolidating, False otherwise.
    """
    if len(highs) < lookback or atr.dropna().empty:
        return False

    recent_highs = highs.iloc[-lookback:]
    recent_lows = lows.iloc[-lookback:]
    current_atr = float(atr.dropna().iloc[-1])

    price_range = float(recent_highs.max()) - float(recent_lows.min())
    threshold = current_atr * lookback * atr_threshold

    return price_range < threshold


def find_support_resistance(
    highs: pd.Series,
    lows: pd.Series,
    lookback: int = 100,
    tolerance: float = 0.002,
) -> tuple[list[float], list[float]]:
    """Identify key support and resistance levels via price clustering.

    Args:
        highs:     Series of high prices.
        lows:      Series of low prices.
        lookback:  Number of recent bars to evaluate.
        tolerance: Percentage tolerance for clustering nearby levels.

    Returns:
        Tuple of (resistance_levels, support_levels) sorted descending/ascending.
    """
    if len(highs) < 10:
        return [], []

    recent_highs = list(highs.iloc[-lookback:])
    recent_lows = list(lows.iloc[-lookback:])

    resistance_levels = _cluster_levels(recent_highs, tolerance)
    support_levels = _cluster_levels(recent_lows, tolerance)

    resistance_levels.sort(reverse=True)
    support_levels.sort()

    return resistance_levels, support_levels


def _cluster_levels(prices: list[float], tolerance: float) -> list[float]:
    """Cluster nearby price levels together."""
    if not prices:
        return []

    sorted_prices = sorted(prices)
    clusters: list[list[float]] = []

    current_cluster = [sorted_prices[0]]

    for price in sorted_prices[1:]:
        ref = current_cluster[0]
        if abs(price - ref) / ref <= tolerance:
            current_cluster.append(price)
        else:
            clusters.append(current_cluster)
            current_cluster = [price]

    clusters.append(current_cluster)

    return [sum(c) / len(c) for c in clusters]
