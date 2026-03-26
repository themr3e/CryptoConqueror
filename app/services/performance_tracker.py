"""Rolling strategy performance metric calculator.

Recalculates 7-day and 30-day rolling performance metrics after each
trade outcome, with results stored in the strategy_performance table.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outcome import Outcome
from app.models.signal import Signal
from app.models.strategy_performance import StrategyPerformance


class PerformanceTracker:
    """Recalculates rolling performance metrics for strategies."""

    async def recalculate_for_strategy(
        self, session: AsyncSession, strategy_id: int
    ) -> None:
        """Recalculate 7d and 30d metrics for the given strategy."""
        for period_days, period_label in [(7, "7d"), (30, "30d")]:
            try:
                await self._compute_and_upsert(session, strategy_id, period_days, period_label)
            except Exception:
                logger.exception(
                    "PerformanceTracker: error computing {}d metrics for strategy_id={}",
                    period_days,
                    strategy_id,
                )

    async def _compute_and_upsert(
        self,
        session: AsyncSession,
        strategy_id: int,
        period_days: int,
        period_label: str,
    ) -> None:
        """Compute metrics for a rolling window and upsert the result."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

        # Query outcomes joined with signals for this strategy in the window
        stmt = (
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(
                and_(
                    Signal.strategy_id == strategy_id,
                    Outcome.created_at >= cutoff,
                )
            )
        )
        result = await session.execute(stmt)
        rows = result.all()

        if not rows:
            return

        total = len(rows)
        wins = sum(1 for o, s in rows if o.result in ("tp1_hit", "tp2_hit"))
        win_rate = Decimal(str(round(wins / total, 4))) if total > 0 else Decimal("0")

        gross_profit = sum(float(o.pnl_pips) for o, s in rows if float(o.pnl_pips) > 0)
        gross_loss = abs(sum(float(o.pnl_pips) for o, s in rows if float(o.pnl_pips) < 0))

        if gross_loss == 0:
            profit_factor = Decimal("9999.9999") if gross_profit > 0 else Decimal("0")
        else:
            pf = min(gross_profit / gross_loss, 9999.9999)
            profit_factor = Decimal(str(round(pf, 4)))

        rr_values = [float(s.risk_reward) for o, s in rows]
        avg_rr = Decimal(str(round(sum(rr_values) / len(rr_values), 4))) if rr_values else Decimal("0")

        is_degraded = float(profit_factor) < 1.0

        await self._upsert_performance(
            session,
            strategy_id=strategy_id,
            period=period_label,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_rr=avg_rr,
            total_signals=total,
            is_degraded=is_degraded,
        )

    async def _upsert_performance(
        self,
        session: AsyncSession,
        strategy_id: int,
        period: str,
        win_rate: Decimal,
        profit_factor: Decimal,
        avg_rr: Decimal,
        total_signals: int,
        is_degraded: bool,
    ) -> None:
        """Update or create a StrategyPerformance row."""
        stmt = select(StrategyPerformance).where(
            and_(
                StrategyPerformance.strategy_id == strategy_id,
                StrategyPerformance.period == period,
            )
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.win_rate = win_rate
            existing.profit_factor = profit_factor
            existing.avg_rr = avg_rr
            existing.total_signals = total_signals
            existing.is_degraded = is_degraded
            existing.calculated_at = datetime.now(timezone.utc)
        else:
            perf = StrategyPerformance(
                strategy_id=strategy_id,
                period=period,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_rr=avg_rr,
                total_signals=total_signals,
                is_degraded=is_degraded,
            )
            session.add(perf)

        await session.commit()
        logger.info(
            "PerformanceTracker: upserted {} metrics for strategy_id={} "
            "(wr={}, pf={}, total={})",
            period,
            strategy_id,
            win_rate,
            profit_factor,
            total_signals,
        )
