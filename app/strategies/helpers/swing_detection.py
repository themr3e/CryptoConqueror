"""Swing high / swing low detection utilities.

Used by liquidity sweep and other strategies to identify structural pivot points.
"""

from __future__ import annotations

import pandas as pd


def find_swing_highs(
    highs: pd.Series,
    order: int = 5,
) -> pd.Series:
    """Identify swing high indices.

    A swing high at index ``i`` requires that ``highs[i]`` is greater than
    all ``order`` bars on each side.

    Args:
        highs: Series of high prices.
        order: Number of bars to look either side (default 5).

    Returns:
        Boolean pd.Series — True at swing high positions.
    """
    n = len(highs)
    is_swing = pd.Series(False, index=highs.index)

    for i in range(order, n - order):
        window_left = highs.iloc[i - order: i]
        window_right = highs.iloc[i + 1: i + order + 1]
        if highs.iloc[i] > window_left.max() and highs.iloc[i] > window_right.max():
            is_swing.iloc[i] = True

    return is_swing


def find_swing_lows(
    lows: pd.Series,
    order: int = 5,
) -> pd.Series:
    """Identify swing low indices.

    Args:
        lows:  Series of low prices.
        order: Number of bars to look either side (default 5).

    Returns:
        Boolean pd.Series — True at swing low positions.
    """
    n = len(lows)
    is_swing = pd.Series(False, index=lows.index)

    for i in range(order, n - order):
        window_left = lows.iloc[i - order: i]
        window_right = lows.iloc[i + 1: i + order + 1]
        if lows.iloc[i] < window_left.min() and lows.iloc[i] < window_right.min():
            is_swing.iloc[i] = True

    return is_swing


def get_recent_swing_high(
    highs: pd.Series,
    order: int = 5,
    lookback: int = 50,
) -> float | None:
    """Return the most recent swing high value within ``lookback`` bars.

    Args:
        highs:    Series of high prices.
        order:    Pivot detection order.
        lookback: How many recent bars to search.

    Returns:
        Most recent swing high price, or None if none found.
    """
    recent = highs.iloc[-lookback:] if len(highs) >= lookback else highs
    swings = find_swing_highs(recent, order=order)
    swing_highs = recent[swings]
    if swing_highs.empty:
        return None
    return float(swing_highs.iloc[-1])


def get_recent_swing_low(
    lows: pd.Series,
    order: int = 5,
    lookback: int = 50,
) -> float | None:
    """Return the most recent swing low value within ``lookback`` bars.

    Args:
        lows:     Series of low prices.
        order:    Pivot detection order.
        lookback: How many recent bars to search.

    Returns:
        Most recent swing low price, or None if none found.
    """
    recent = lows.iloc[-lookback:] if len(lows) >= lookback else lows
    swings = find_swing_lows(recent, order=order)
    swing_lows = recent[swings]
    if swing_lows.empty:
        return None
    return float(swing_lows.iloc[-1])
