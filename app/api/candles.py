"""Candles query API endpoints.

Provides REST endpoints for querying stored OHLCV candle data.
"""

from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.candle import Candle
from app.schemas.candle import CandleResponse

router = APIRouter(prefix="/candles", tags=["candles"])


class TimeframeEnum(str, Enum):
    """Valid timeframe values."""

    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


@router.get("/{timeframe}", response_model=list[CandleResponse])
async def get_candles(
    timeframe: TimeframeEnum,
    symbol: str = Query(default="BTCUSDT", description="Symbol to query candles for"),
    limit: int = Query(default=100, ge=1, le=5000),
    start: datetime | None = Query(default=None, description="Filter candles from this timestamp (inclusive)"),
    end: datetime | None = Query(default=None, description="Filter candles until this timestamp (inclusive)"),
    session: AsyncSession = Depends(get_session),
) -> list[CandleResponse]:
    """Query stored candles for a given symbol and timeframe."""
    query = (
        select(Candle)
        .where(Candle.symbol == symbol.upper())
        .where(Candle.timeframe == timeframe.value)
    )

    if start is not None:
        query = query.where(Candle.timestamp >= start)
    if end is not None:
        query = query.where(Candle.timestamp <= end)

    query = query.order_by(Candle.timestamp.desc()).limit(limit)

    result = await session.execute(query)
    candles = result.scalars().all()

    return [CandleResponse.model_validate(c) for c in candles]
