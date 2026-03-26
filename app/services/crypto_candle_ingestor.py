"""Crypto futures candle ingestor using Binance Futures REST API.

Fetches OHLCV klines from ``https://fapi.binance.com`` and upserts them
into the shared ``candles`` table with ``source="binance_futures"``.

No API key is required for public market data endpoints.

Supported timeframes (mapped to Binance interval strings):
    M15 → 15m,  H1 → 1h,  H4 → 4h,  D1 → 1d
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any

import httpx
from loguru import logger
from sqlalchemy import and_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential

from app.models.candle import Candle

# Binance Futures base URL (public, no auth required for market data)
_BASE_URL = "https://fapi.binance.com/fapi/v1"

_INTERVAL_MAP: dict[str, str] = {
    "M15": "15m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}

# Binance returns up to 1500 klines per request
_MAX_LIMIT = 1500


class CryptoCandleIngestor:
    """Fetches and stores Binance Futures klines for crypto symbols."""

    def __init__(self, timeout: float = 20.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def fetch_and_store(
        self,
        session: AsyncSession,
        symbol: str,
        timeframe: str,
        limit: int = _MAX_LIMIT,
    ) -> int:
        """Incremental fetch: only pulls candles newer than latest stored.

        Args:
            session:   Async DB session.
            symbol:    Binance symbol, e.g. ``"BTCUSDT"``.
            timeframe: App timeframe key (``"M15"``, ``"H1"``, ``"H4"``, ``"D1"``).
            limit:     Max candles to fetch in a single request.

        Returns:
            Number of candles stored (inserted or updated).
        """
        latest = await self.get_latest_timestamp(session, symbol, timeframe)
        start_ms: int | None = None
        if latest is not None:
            # Start just after the last known candle
            start_ms = int(latest.timestamp() * 1000) + 1

        candles = await self._fetch_klines(symbol, timeframe, limit=limit, start_ms=start_ms)
        if not candles:
            logger.debug("CryptoCandleIngestor: no new candles for {} {}", symbol, timeframe)
            return 0

        return await self.upsert_candles(session, candles)

    async def get_latest_timestamp(
        self,
        session: AsyncSession,
        symbol: str,
        timeframe: str,
    ) -> datetime | None:
        """Return the timestamp of the most recent stored candle for this symbol/tf."""
        stmt = (
            select(Candle.timestamp)
            .where(and_(Candle.symbol == symbol, Candle.timeframe == timeframe))
            .order_by(Candle.timestamp.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_candles(
        self,
        session: AsyncSession,
        candles: list[dict[str, Any]],
    ) -> int:
        """Upsert candle rows using PostgreSQL ON CONFLICT DO UPDATE.

        Args:
            session: Async DB session.
            candles: List of candle dicts from ``_fetch_klines``.

        Returns:
            Number of rows affected.
        """
        if not candles:
            return 0

        stmt = pg_insert(Candle).values(candles)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_candle_identity",
            set_={
                "open":   stmt.excluded.open,
                "high":   stmt.excluded.high,
                "low":    stmt.excluded.low,
                "close":  stmt.excluded.close,
                "volume": stmt.excluded.volume,
                "source": stmt.excluded.source,
            },
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount

    async def get_mark_price(self, symbol: str) -> float | None:
        """Fetch current mark price from Binance premiumIndex endpoint.

        Uses mark price (not last trade) to avoid liquidation wick distortion.

        Args:
            symbol: Binance symbol, e.g. ``"BTCUSDT"``.

        Returns:
            Mark price as float, or None on failure.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{_BASE_URL}/premiumIndex",
                    params={"symbol": symbol},
                )
                resp.raise_for_status()
                data = resp.json()
                return float(data["markPrice"])
        except Exception:
            logger.opt(exception=True).warning("Failed to fetch mark price for {}", symbol)
            return None

    async def detect_gaps(
        self,
        session: AsyncSession,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[datetime]:
        """Detect missing candle timestamps between start and end.

        Crypto trades 24/7 — no weekend filter is applied.

        Args:
            session:   Async DB session.
            symbol:    Symbol to check.
            timeframe: Timeframe key.
            start:     Start of range (UTC-aware).
            end:       End of range (UTC-aware).

        Returns:
            List of missing timestamps.
        """
        interval_minutes = _timeframe_minutes(timeframe)

        result = await session.execute(
            text("""
                SELECT gs
                FROM generate_series(
                    :start::timestamptz,
                    :end::timestamptz,
                    (:interval_minutes || ' minutes')::interval
                ) AS gs
                WHERE gs NOT IN (
                    SELECT timestamp FROM candles
                    WHERE symbol = :symbol AND timeframe = :timeframe
                      AND timestamp BETWEEN :start AND :end
                )
                ORDER BY gs
            """),
            {
                "start": start,
                "end": end,
                "interval_minutes": interval_minutes,
                "symbol": symbol,
                "timeframe": timeframe,
            },
        )
        return [row[0] for row in result.fetchall()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=16),
        reraise=True,
    )
    async def _fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int = _MAX_LIMIT,
        start_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch klines from Binance Futures API with retry.

        Args:
            symbol:    e.g. ``"BTCUSDT"``.
            timeframe: App timeframe key.
            limit:     Number of candles.
            start_ms:  Optional start time in milliseconds (UTC epoch).

        Returns:
            List of candle dicts ready for ``upsert_candles``.
        """
        interval = _INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, _MAX_LIMIT),
        }
        if start_ms is not None:
            params["startTime"] = start_ms

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{_BASE_URL}/klines", params=params)
            resp.raise_for_status()
            raw: list[list[Any]] = resp.json()

        candles: list[dict[str, Any]] = []
        for row in raw:
            # Binance kline format:
            # [0]open_time, [1]open, [2]high, [3]low, [4]close, [5]volume,
            # [6]close_time, [7]quote_asset_volume, [8]count, ...
            open_time_ms: int = row[0]
            ts = datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc)

            candles.append({
                "symbol":    symbol,
                "timeframe": timeframe,
                "timestamp": ts,
                "open":      Decimal(str(row[1])),
                "high":      Decimal(str(row[2])),
                "low":       Decimal(str(row[3])),
                "close":     Decimal(str(row[4])),
                "volume":    Decimal(str(row[5])),
                "source":    "binance_futures",
            })

        logger.debug(
            "CryptoCandleIngestor: fetched {} candles for {} {}",
            len(candles), symbol, timeframe,
        )
        return candles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timeframe_minutes(timeframe: str) -> int:
    """Convert timeframe key to integer minutes."""
    mapping = {"M15": 15, "H1": 60, "H4": 240, "D1": 1440}
    return mapping.get(timeframe, 60)
