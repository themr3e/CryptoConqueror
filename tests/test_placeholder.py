"""Placeholder test module.

Ensures the test suite can be discovered and run without errors.
Real tests should be added here as the system evolves.
"""


def test_import_strategies() -> None:
    """Smoke test: all strategies can be imported."""
    import app.strategies  # noqa: F401 — triggers self-registration
    from app.strategies.base import BaseStrategy

    registry = BaseStrategy.get_registry()
    assert len(registry) == 4, f"Expected 4 strategies, got {len(registry)}"
    assert "liquidity_sweep" in registry
    assert "trend_continuation" in registry
    assert "breakout_expansion" in registry
    assert "ema_momentum" in registry


def test_session_filter() -> None:
    """Smoke test: session filter returns a list."""
    from app.strategies.helpers.session_filter import get_active_sessions

    sessions = get_active_sessions()
    assert isinstance(sessions, list)


def test_indicators_atr() -> None:
    """Smoke test: ATR computation returns correct length."""
    import pandas as pd
    from app.strategies.helpers.indicators import compute_atr

    highs = pd.Series([1.1, 1.2, 1.3, 1.2, 1.1, 1.4, 1.5, 1.3, 1.2, 1.1,
                       1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.6, 1.5, 1.4, 1.3])
    lows  = pd.Series([1.0, 1.1, 1.2, 1.1, 1.0, 1.3, 1.4, 1.2, 1.1, 1.0,
                       1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3, 1.2])
    closes = pd.Series([1.05, 1.15, 1.25, 1.15, 1.05, 1.35, 1.45, 1.25, 1.15,
                        1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.55, 1.45,
                        1.35, 1.25])

    atr = compute_atr(highs, lows, closes, length=14)
    assert len(atr) == len(highs)
    assert atr.dropna().iloc[-1] > 0
