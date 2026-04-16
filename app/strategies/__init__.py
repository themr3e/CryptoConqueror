"""Strategy registry — import all concrete strategies so they self-register."""

from app.strategies.crypto_momentum import CryptoMomentumStrategy
from app.strategies.crypto_breakout import CryptoBreakoutStrategy
from app.strategies.crypto_slc import CryptoSLCStrategy

__all__ = [
    "CryptoMomentumStrategy",
    "CryptoBreakoutStrategy",
    "CryptoSLCStrategy",
]
