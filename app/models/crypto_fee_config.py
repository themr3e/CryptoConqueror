"""Crypto futures fee configuration model.

Stores per-symbol fee parameters for Binance Futures.
Seeded via Alembic migration; adjustable at runtime.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CryptoFeeConfig(Base):
    __tablename__ = "crypto_fee_configs"

    __table_args__ = (
        UniqueConstraint("symbol", name="uq_crypto_fee_symbol"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20))               # e.g. "BTCUSDT"
    taker_fee_rate: Mapped[Decimal] = mapped_column(Numeric(8, 6))  # e.g. 0.000450
    maker_fee_rate: Mapped[Decimal] = mapped_column(Numeric(8, 6))  # e.g. 0.000200
    funding_rate_8h: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 8), nullable=True
    )  # latest 8h funding rate (informational)
    tick_size: Mapped[Decimal] = mapped_column(Numeric(18, 8))     # minimum price increment
    lot_size: Mapped[Decimal] = mapped_column(Numeric(18, 8))      # minimum quantity step
    leverage_max: Mapped[int] = mapped_column(Integer, default=20)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )
