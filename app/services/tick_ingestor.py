"""Tick-level data ingestor for the footprint / order-flow engine.

For a "hot" subset of symbols (BTC/ETH/SOL) we pull :class:`aggTrades`
from Binance Futures for full tick-by-tick fidelity. For the remaining
symbols we fall back to 1-second klines and approximate trade side via
``sign(close - open)`` — the same approximation used by the reference
TradingView indicator when 1s granularity is selected.

Each pulled trade is persisted to :class:`RawTrade`. The 1-minute
rollup into :class:`FootprintBar` is performed by
:func:`rollup_minute` which delegates the math to
:mod:`footprint_analyzer`.

Exports:
    TickIngestor -- ingests aggTrades / klines_1s into raw_trades
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.footprint_bar import FootprintBar
from app.models.raw_trade import RawTrade
from app.services.binance_executor import BinanceExecutor
from app.services.footprint_analyzer import (
    build_profile,
    compute_poc_va,
    count_stacked,
    derive_tick_size,
    find_imbalances,
    rows_to_trades,
)


# Symbols that use full aggTrades (the rest use 1s klines).
_TICK_ACCURATE_SYMBOLS: set[str] = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


class TickIngestor:
    """Pulls tick-level data from Binance and stores it in raw_trades."""

    def __init__(self, executor: BinanceExecutor | None = None) -> None:
        self._executor = executor or BinanceExecutor()

    @staticmethod
    def is_tick_accurate(symbol: str) -> bool:
        """Return True if we fetch aggTrades for this symbol (vs. 1s klines)."""
        return symbol in _TICK_ACCURATE_SYMBOLS

    async def ingest_symbol(
        self,
        session: AsyncSession,
        symbol: str,
        lookback_minutes: int = 2,
    ) -> int:
        """Ingest the last ``lookback_minutes`` of trades for a symbol.

        Returns the number of raw_trades rows inserted (or upserted).
        """
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - lookback_minutes * 60_000

        if self.is_tick_accurate(symbol):
            rows = await self._ingest_agg_trades(session, symbol, start_ms, end_ms)
        else:
            rows = await self._ingest_1s_klines(session, symbol, start_ms, end_ms)
        return rows

    # ------------------------------------------------------------------
    # aggTrades path (BTC/ETH/SOL)
    # ------------------------------------------------------------------

    async def _ingest_agg_trades(
        self,
        session: AsyncSession,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> int:
        """Paginated fetch of aggTrades by time window, then upsert."""
        # Binance caps the aggTrades time range to 1 hour per request.
        # Our lookback window is short (≤2 min) so one call is enough.
        trades = await self._executor.fetch_agg_trades(
            symbol, start_ms=start_ms, end_ms=end_ms, limit=1000,
        )
        if not trades:
            return 0

        rows = [self._agg_trade_to_row(symbol, t) for t in trades]
        return await self._upsert_raw_trades(session, rows)

    @staticmethod
    def _agg_trade_to_row(symbol: str, t: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "trade_id": int(t["a"]),
            "price": Decimal(str(t["p"])),
            "qty": Decimal(str(t["q"])),
            "is_buyer_maker": bool(t["m"]),
            "ts": datetime.fromtimestamp(int(t["T"]) / 1000, tz=timezone.utc),
            "source": "agg_trades",
        }

    # ------------------------------------------------------------------
    # 1s-klines fallback path (every other symbol)
    # ------------------------------------------------------------------

    async def _ingest_1s_klines(
        self,
        session: AsyncSession,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> int:
        """Fetch 1s klines, convert each kline into one synthetic trade row."""
        klines = await self._executor.fetch_klines_1s(
            symbol, start_ms=start_ms, end_ms=end_ms, limit=1000,
        )
        if not klines:
            return 0

        rows: list[dict[str, Any]] = []
        for k in klines:
            # Binance kline layout: [openTime, open, high, low, close, volume, closeTime, ...]
            open_price = float(k[1])
            close_price = float(k[4])
            volume = float(k[5])
            if volume <= 0:
                continue
            open_ms = int(k[0])
            # sign(close - open): matches Pine 1m/1s mode classification.
            # is_buyer_maker == True means taker was SELLER (aggressive sell),
            # which maps to a DOWN candle (close < open).
            is_buyer_maker = close_price < open_price
            # For close == open (doji) fall back to previous direction via
            # is_buyer_maker=False — treated as zero-delta in the analyzer
            # because sign(0) == 0 there. See `footprint_analyzer.sign_of_kline`.
            rows.append({
                "symbol": symbol,
                # Synthetic trade_id: open_ms uniquely identifies this 1s bar.
                "trade_id": open_ms,
                "price": Decimal(str(close_price)),
                "qty": Decimal(str(volume)),
                "is_buyer_maker": is_buyer_maker,
                "ts": datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc),
                "source": "klines_1s",
            })
        return await self._upsert_raw_trades(session, rows)

    # ------------------------------------------------------------------
    # Upsert (idempotent on symbol + trade_id)
    # ------------------------------------------------------------------

    @staticmethod
    async def _upsert_raw_trades(
        session: AsyncSession,
        rows: list[dict[str, Any]],
    ) -> int:
        if not rows:
            return 0
        stmt = pg_insert(RawTrade).values(rows)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_raw_trade_identity")
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount or 0

    # ------------------------------------------------------------------
    # Cursor helper — used by jobs to avoid redundant re-fetches
    # ------------------------------------------------------------------

    @staticmethod
    async def latest_ts(session: AsyncSession, symbol: str) -> datetime | None:
        """Return the most recent raw_trade timestamp we have for ``symbol``."""
        stmt = select(func.max(RawTrade.ts)).where(RawTrade.symbol == symbol)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Retention: only keep enough trades to rollup the previous minute
    # ------------------------------------------------------------------

    @staticmethod
    async def has_recent_bar(
        session: AsyncSession,
        symbol: str,
        minute_ts: datetime,
    ) -> bool:
        """Return True when a FootprintBar already exists for (symbol, ts)."""
        stmt = (
            select(FootprintBar.id)
            .where(FootprintBar.symbol == symbol, FootprintBar.ts == minute_ts)
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None


# ---------------------------------------------------------------------------
# Minute rollup — reads raw_trades, writes footprint_bars
# ---------------------------------------------------------------------------

def _floor_to_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


async def rollup_minute(
    session: AsyncSession,
    symbol: str,
    minute_ts: datetime,
    fallback_tick_size: float = 0.01,
) -> FootprintBar | None:
    """Roll up raw_trades for ``(symbol, minute_ts .. minute_ts+60s)`` into
    one :class:`FootprintBar` row.

    Skips (returns None) when the minute already exists or has no trades.
    Computes cumulative delta by adding this minute's delta to the previous
    bar's cum_delta.
    """
    minute_ts = _floor_to_minute(minute_ts)
    next_ts = minute_ts + timedelta(minutes=1)

    # Skip if we already rolled up this minute.
    existing = await session.execute(
        select(FootprintBar.id)
        .where(FootprintBar.symbol == symbol, FootprintBar.ts == minute_ts)
        .limit(1)
    )
    if existing.scalar_one_or_none() is not None:
        return None

    # Load trades for this minute window.
    stmt = (
        select(RawTrade)
        .where(
            RawTrade.symbol == symbol,
            RawTrade.ts >= minute_ts,
            RawTrade.ts < next_ts,
        )
        .order_by(RawTrade.ts.asc())
    )
    res = await session.execute(stmt)
    rows = res.scalars().all()
    if not rows:
        return None

    trades = rows_to_trades(rows)
    if not trades:
        return None

    high = max(t.price for t in trades)
    low = min(t.price for t in trades)
    tick_size = derive_tick_size(high, low, fallback_tick_size)

    profile = build_profile(trades, tick_size)
    if profile is None:
        return None

    compute_poc_va(profile, va_pct=0.70)
    imbalances = find_imbalances(profile, threshold=0.70)
    count, side = count_stacked(imbalances, min_stack=3)

    # Cumulative delta = previous bar's cum_delta + this bar's delta.
    prev_stmt = (
        select(FootprintBar.cum_delta)
        .where(FootprintBar.symbol == symbol, FootprintBar.ts < minute_ts)
        .order_by(FootprintBar.ts.desc())
        .limit(1)
    )
    prev_cd_res = await session.execute(prev_stmt)
    prev_cd = prev_cd_res.scalar_one_or_none()
    cum_delta = float(prev_cd or 0.0) + profile.delta

    src = rows[0].source if rows else "agg_trades"

    bar = FootprintBar(
        symbol=symbol,
        ts=minute_ts,
        tick_size=Decimal(str(round(tick_size, 8))),
        levels=[lv.to_dict() for lv in profile.levels if lv.total_vol > 0],
        poc=Decimal(str(round(profile.poc, 8))),
        vah=Decimal(str(round(profile.vah, 8))),
        val=Decimal(str(round(profile.val, 8))),
        total_vol=Decimal(str(round(profile.total_vol, 8))),
        buy_vol=Decimal(str(round(profile.buy_vol, 8))),
        sell_vol=Decimal(str(round(profile.sell_vol, 8))),
        delta=Decimal(str(round(profile.delta, 8))),
        cum_delta=Decimal(str(round(cum_delta, 8))),
        stacked_imb_count=count,
        stacked_imb_side=side,
        high=Decimal(str(round(profile.high, 8))),
        low=Decimal(str(round(profile.low, 8))),
        close=Decimal(str(round(profile.close, 8))),
        source=src,
    )
    session.add(bar)
    await session.commit()
    return bar
