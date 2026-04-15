"""Unit tests for the footprint analyzer."""

from __future__ import annotations

from app.services.footprint_analyzer import (
    Trade,
    build_profile,
    compute_poc_va,
    count_stacked,
    cumulative_delta_divergence,
    derive_tick_size,
    find_imbalances,
)


def _mk(price: float, qty: float, sell: bool) -> Trade:
    """Build a Trade. ``sell=True`` → aggressive SELL (is_buyer_maker=True)."""
    return Trade(price=price, qty=qty, is_buyer_maker=sell)


def test_build_profile_buckets_trades_into_levels() -> None:
    trades = [
        _mk(100.0, 5, sell=False),   # +5 buy   → level 100
        _mk(100.4, 3, sell=True),    # -3 sell  → level 100 (same bucket, tick=1)
        _mk(101.0, 10, sell=False),  # +10 buy  → level 101
        _mk(102.0, 4, sell=True),    # -4 sell  → level 102
    ]
    profile = build_profile(trades, tick_size=1.0)
    assert profile is not None
    assert profile.high == 102.0
    assert profile.low == 100.0
    assert profile.total_vol == 22.0
    assert profile.buy_vol == 15.0
    assert profile.sell_vol == 7.0
    assert profile.delta == 8.0


def test_poc_is_highest_volume_level() -> None:
    trades = [
        _mk(100.0, 1, sell=False),
        _mk(101.0, 20, sell=False),   # heaviest level
        _mk(101.0, 5, sell=True),
        _mk(102.0, 2, sell=True),
    ]
    profile = build_profile(trades, tick_size=1.0)
    assert profile is not None
    poc, vah, val = compute_poc_va(profile, va_pct=0.70)
    assert poc == 101.0
    # Value area should include the POC level and widen until ≥70%
    assert val <= 101.0 <= vah


def test_imbalance_detection_and_stacked_count() -> None:
    # Three consecutive levels with pure BUY volume → stacked imbalance
    trades = [
        _mk(100.0, 10, sell=False),
        _mk(101.0, 10, sell=False),
        _mk(102.0, 10, sell=False),
        _mk(103.0, 1, sell=True),  # adds a non-imbalance level after
    ]
    profile = build_profile(trades, tick_size=1.0)
    assert profile is not None
    compute_poc_va(profile)
    imbs = find_imbalances(profile, threshold=0.70)
    assert len(imbs) >= 3
    assert all(side == "BUY" for _, side in imbs[:3])

    count, side = count_stacked(imbs, min_stack=3)
    assert count >= 3
    assert side == "BUY"


def test_stacked_short_of_threshold_returns_zero() -> None:
    imbs = [(0, "BUY"), (1, "BUY")]  # only 2 consecutive
    count, side = count_stacked(imbs, min_stack=3)
    assert count == 0
    assert side is None


def test_stacked_breaks_on_direction_flip() -> None:
    imbs = [(0, "BUY"), (1, "BUY"), (2, "SELL"), (3, "BUY")]
    count, side = count_stacked(imbs, min_stack=3)
    # No run of 3 in either direction
    assert count == 0


def test_stacked_breaks_on_gap() -> None:
    imbs = [(0, "BUY"), (2, "BUY"), (3, "BUY")]  # gap at index 1
    count, side = count_stacked(imbs, min_stack=3)
    # longest consecutive run of BUY is [2,3] → length 2
    assert count == 0


def test_bearish_divergence_detected() -> None:
    # First half: price low, cum_delta high.
    # Second half: price higher high, cum_delta lower high → bearish div.
    closes =    [100, 101, 102, 103, 104, 110, 111, 112, 113, 115]
    cum_delta = [100, 110, 120, 130, 140,  90,  95,  85,  80,  70]
    div = cumulative_delta_divergence(closes, cum_delta, lookback=10)
    assert div == "bearish"


def test_bullish_divergence_detected() -> None:
    closes =    [110, 109, 108, 107, 106, 105, 104, 103, 102, 100]
    cum_delta = [-100, -110, -120, -130, -140, -90, -80, -70, -60, -50]
    div = cumulative_delta_divergence(closes, cum_delta, lookback=10)
    assert div == "bullish"


def test_no_divergence_when_trend_agrees() -> None:
    closes =    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    cum_delta = [ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100]
    div = cumulative_delta_divergence(closes, cum_delta, lookback=10)
    assert div is None


def test_derive_tick_size_fallback_when_flat_range() -> None:
    ts = derive_tick_size(high=100.0, low=100.0, fallback=0.01)
    assert ts == 0.01


def test_derive_tick_size_scales_with_range() -> None:
    # 1-point range / 20 = 0.05, which exceeds fallback=0.01
    ts = derive_tick_size(high=101.0, low=100.0, fallback=0.01)
    assert ts == 0.05


def test_build_profile_empty_trades_returns_none() -> None:
    assert build_profile([], tick_size=1.0) is None
