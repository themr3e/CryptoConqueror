"""Claude autonomous trading decision log."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class ClaudeDecision(Base):
    """Records every decision Claude makes with full reasoning and outcome."""

    __tablename__ = "claude_decisions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(20), nullable=False)  # open_long, open_short, close, hold
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    position_size: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    executed: Mapped[bool] = mapped_column(default=False)
    execution_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    daily_pnl_at_decision: Mapped[float | None] = mapped_column(Numeric(18, 8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
