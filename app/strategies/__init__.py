"""Strategy registry — import all concrete strategies so they self-register."""

from app.strategies.crypto_momentum import CryptoMomentumStrategy
from app.strategies.crypto_breakout import CryptoBreakoutStrategy

__all__ = [
    "CryptoMomentumStrategy",
    "CryptoBreakoutStrategy",
]
