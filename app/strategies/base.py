"""Base strategy class and CandidateSignal schema.

All concrete strategies inherit from BaseStrategy and must implement
``generate_signals()``.

Exports:
    BaseStrategy    -- abstract base class
    CandidateSignal -- pydantic model for unvalidated signal candidates
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict


class InsufficientDataError(Exception):
    """Raised when there is not enough candle data to run a strategy."""


class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class CandidateSignal(BaseModel):
    """A raw signal candidate produced by a strategy before validation."""

    model_config = ConfigDict(use_enum_values=False)

    strategy_name: str
    symbol: str
    timeframe: str
    direction: SignalDirection
    entry_price: Decimal
    stop_loss: Decimal
    take_profit_1: Decimal
    take_profit_2: Decimal | None = None
    risk_reward: Decimal
    confidence: Decimal
    reasoning: str
    session: str = "unknown"
    timestamp: datetime | None = None   # candle timestamp that triggered the signal


_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {}


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must define:
        - ``NAME``           : unique strategy identifier
        - ``DEFAULT_PARAMS`` : dict of default parameter values
        - ``generate_signals(candles)``: returns list of CandidateSignal
    """

    NAME: ClassVar[str]
    DEFAULT_PARAMS: ClassVar[dict[str, float]]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        if params is not None:
            self.params: dict[str, float] = {**self.DEFAULT_PARAMS, **params}
        else:
            self.params = dict(self.DEFAULT_PARAMS)

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register concrete strategy subclasses."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "NAME") and cls.NAME:
            _STRATEGY_REGISTRY[cls.NAME] = cls

    @abstractmethod
    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        """Generate signal candidates from OHLCV candle data.

        Args:
            candles: DataFrame with columns: open, high, low, close, volume.
                     Indexed by datetime (UTC-aware).

        Returns:
            List of CandidateSignal instances (may be empty).
        """

    @classmethod
    def get_registry(cls) -> dict[str, type[BaseStrategy]]:
        """Return the strategy registry."""
        return _STRATEGY_REGISTRY

    def _to_decimal(self, value: float, places: int = 5) -> Decimal:
        """Round float to Decimal with given decimal places."""
        return Decimal(str(round(value, places)))
