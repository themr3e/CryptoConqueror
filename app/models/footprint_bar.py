"""Footprint bar — per-minute rolled-up order-flow profile.

For each symbol + minute we persist the computed footprint: a volume
profile bucketed into ATR-derived tick levels, point-of-control (POC),
value area (70%), cumulative delta, and stacked-imbalance count/side.

This is the table read by :class:`ClaudeTradeAgent` and by
:class:`FootprintImbalanceStrategy` — raw ``raw_trades`` are only
consumed by the rollup job.

The ``levels`` column is a JSONB array of
``{"price": float, "buy_vol": float, "sell_vol": float, "delta": float}``
dicts sorted ascending by price.

Retention: 7 days (enforced by :class:`DataRetentionService`).
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import BigInteger, DateTime, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class FootprintBar(Base):
    __tablename__ = "footprint_bars"

    __table_args__ = (
        UniqueConstraint("symbol", "ts", name="uq_footprint_bar_identity"),
        Index("idx_footprint_bars_symbol_ts", "symbol", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # minute boundary (UTC)
    tick_size: Mapped[Decimal] = mapped_column(Numeric(18, 8))

    # Raw profile — array of {price, buy_vol, sell_vol, delta}
    levels: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)

    # Derived scalars (kept denormalised for fast Claude prompts)
    poc: Mapped[Decimal] = mapped_column(Numeric(18, 8))              # price with max total volume
    vah: Mapped[Decimal] = mapped_column(Numeric(18, 8))              # value-area high
    val: Mapped[Decimal] = mapped_column(Numeric(18, 8))              # value-area low
    total_vol: Mapped[Decimal] = mapped_column(Numeric(24, 8))
    buy_vol: Mapped[Decimal] = mapped_column(Numeric(24, 8))
    sell_vol: Mapped[Decimal] = mapped_column(Numeric(24, 8))
    delta: Mapped[Decimal] = mapped_column(Numeric(24, 8))            # buy_vol − sell_vol (this bar)
    cum_delta: Mapped[Decimal] = mapped_column(Numeric(24, 8))        # rolling cumulative delta

    stacked_imb_count: Mapped[int] = mapped_column(Integer, default=0)
    stacked_imb_side: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)  # "BUY" | "SELL" | None

    high: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    low: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    close: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    source: Mapped[str] = mapped_column(String(20), default="agg_trades")
