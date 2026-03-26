"""Trade order tracking model.

Records every order placed on Binance Futures (testnet or live).
Each signal can have up to 3 linked orders: entry, stop-loss, take-profit.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class TradeOrder(Base):
    __tablename__ = "trade_orders"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Link back to the signal that triggered this order
    signal_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("signals.id"))

    # Binance-assigned order ID (used for cancellation / status checks)
    broker_order_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # "entry" | "stop_loss" | "take_profit_1" | "take_profit_2"
    order_role: Mapped[str] = mapped_column(String(20))

    symbol: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(5))      # "BUY" | "SELL"
    order_type: Mapped[str] = mapped_column(String(30))  # "MARKET" | "STOP_MARKET" | "TAKE_PROFIT_MARKET"

    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 8))
    price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8), nullable=True)
    stop_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8), nullable=True)

    # "NEW" | "FILLED" | "CANCELED" | "REJECTED" | "ERROR"
    status: Mapped[str] = mapped_column(String(20), default="NEW")

    # Full JSON response from Binance stored for debugging
    raw_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # "testnet" | "mainnet"
    environment: Mapped[str] = mapped_column(String(10), default="testnet")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )
