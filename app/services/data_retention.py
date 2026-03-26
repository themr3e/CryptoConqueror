"""Data retention service for pruning old candle and backtest data.

Enforces configurable retention policies per timeframe. M15 and H1
candles are pruned after their respective thresholds; H4 and D1 candles
are never pruned. Backtest results older than 180 days are pruned.
Signals and Outcomes are never pruned.

Exports:
    DataRetentionService -- main service class
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.backtest_result import BacktestResult
from app.models.candle import Candle


class DataRetentionService:
    """Prune old candle and backtest data based on configurable retention policies."""

    RETENTION_DAYS: dict[str, int] = {
        "M15": 90,
        "H1": 365,
    }
    BACKTEST_RETENTION_DAYS: int = 180

    async def run(self, session: AsyncSession) -> dict[str, int]:
        """Execute retention policies and return deletion counts."""
        results: dict[str, int] = {}
        now = datetime.now(timezone.utc)

        for timeframe, days in self.RETENTION_DAYS.items():
            cutoff = now - timedelta(days=days)
            stmt = delete(Candle).where(
                Candle.timeframe == timeframe,
                Candle.timestamp < cutoff,
            )
            result = await session.execute(stmt)
            results[f"{timeframe}_candles"] = result.rowcount

            logger.info(
                "data_retention: pruned {count} {tf} candles older than {days}d",
                count=result.rowcount,
                tf=timeframe,
                days=days,
            )

        backtest_cutoff = now - timedelta(days=self.BACKTEST_RETENTION_DAYS)
        stmt = delete(BacktestResult).where(
            BacktestResult.created_at < backtest_cutoff,
        )
        result = await session.execute(stmt)
        results["backtest_results"] = result.rowcount

        logger.info(
            "data_retention: pruned {count} backtest results older than {days}d",
            count=result.rowcount,
            days=self.BACKTEST_RETENTION_DAYS,
        )

        await session.commit()
        return results
