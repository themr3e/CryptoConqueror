"""OHLCV candle data model.

Supports both XAUUSD (Twelve Data) and crypto futures (Binance).
Price columns use Numeric(18, 8) to handle BTC prices and crypto precision.
The ``source`` column distinguishes data origin.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Candle(Base):
    __tablename__ = "candles"

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle_identity"),
        Index("idx_candles_lookup", "symbol", "timeframe", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20))               # e.g. "XAUUSD", "BTCUSDT"
    timeframe: Mapped[str] = mapped_column(String(5))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    open: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    high: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    low: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    close: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    volume: Mapped[Optional[Decimal]] = mapped_column(Numeric(24, 8), nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="twelve_data")  # "twelve_data" | "binance_futures"
