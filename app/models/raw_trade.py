"""Raw aggregated trade data model.

Stores per-trade records for crypto futures pulled from Binance ``aggTrades``
endpoint. Used to build footprint/volume-profile analytics.

Each row corresponds to one aggTrade from Binance (aggregated fills at a
single price level from a single taker order). ``is_buyer_maker`` is the
classification bit: when True the taker was a SELLER (aggressive sell);
when False the taker was a BUYER (aggressive buy).

Retention: 7 days (enforced by :class:`DataRetentionService`).
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, DateTime, Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class RawTrade(Base):
    __tablename__ = "raw_trades"

    __table_args__ = (
        UniqueConstraint("symbol", "trade_id", name="uq_raw_trade_identity"),
        Index("idx_raw_trades_symbol_ts", "symbol", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20))
    trade_id: Mapped[int] = mapped_column(BigInteger)  # Binance aggTradeId
    price: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    qty: Mapped[Decimal] = mapped_column(Numeric(24, 8))
    is_buyer_maker: Mapped[bool] = mapped_column(Boolean)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # trade timestamp (T field)
    source: Mapped[str] = mapped_column(String(20), default="agg_trades")  # "agg_trades" | "klines_1s"
