"""Strategy registry — import all concrete strategies so they self-register."""

from app.strategies.breakout_expansion import BreakoutExpansionStrategy
from app.strategies.ema_momentum import EMAMomentumStrategy
from app.strategies.liquidity_sweep import LiquiditySweepStrategy
from app.strategies.trend_continuation import TrendContinuationStrategy

__all__ = [
    "BreakoutExpansionStrategy",
    "EMAMomentumStrategy",
    "LiquiditySweepStrategy",
    "TrendContinuationStrategy",
]
