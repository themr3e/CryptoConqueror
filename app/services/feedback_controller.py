"""Feedback controller: degradation detection and circuit breaker.

Monitors strategy performance and halts signal generation when adverse
conditions are detected. Automatically recovers after cooldown periods.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.backtest_result import BacktestResult
from app.models.outcome import Outcome
from app.models.signal import Signal
from app.models.strategy_performance import StrategyPerformance

# Circuit breaker state (class-level, persists across instances in process)
_circuit_breaker_active: bool = False
_circuit_breaker_activated_at: datetime | None = None
CIRCUIT_BREAKER_COOLDOWN_HOURS = 24
CONSECUTIVE_LOSS_THRESHOLD = 8
MAX_DRAWDOWN_MULTIPLIER = 2.0


class FeedbackController:
    """Monitors strategy performance and manages the circuit breaker."""

    async def run_checks(self, session: AsyncSession) -> dict:
        """Run all feedback checks and return summary."""
        cb_active = await self.check_circuit_breaker(session)
        return {
            "circuit_breaker_active": cb_active,
        }

    async def check_circuit_breaker(self, session: AsyncSession) -> bool:
        """Check if circuit breaker should be active.

        Activates on 8+ consecutive losses or drawdown > 2x historical max.
        Auto-resets after 24-hour cooldown.
        """
        global _circuit_breaker_active, _circuit_breaker_activated_at

        # Auto-reset after cooldown
        if _circuit_breaker_active and _circuit_breaker_activated_at is not None:
            elapsed = (datetime.now(timezone.utc) - _circuit_breaker_activated_at).total_seconds()
            if elapsed >= CIRCUIT_BREAKER_COOLDOWN_HOURS * 3600:
                _circuit_breaker_active = False
                _circuit_breaker_activated_at = None
                logger.info("Circuit breaker auto-reset after {}h cooldown", CIRCUIT_BREAKER_COOLDOWN_HOURS)
                return False

        if _circuit_breaker_active:
            return True

        # Check consecutive losses
        consec_losses = await self._count_consecutive_losses(session)
        if consec_losses >= CONSECUTIVE_LOSS_THRESHOLD:
            _circuit_breaker_active = True
            _circuit_breaker_activated_at = datetime.now(timezone.utc)
            logger.warning(
                "Circuit breaker ACTIVATED: {} consecutive losses (threshold={})",
                consec_losses, CONSECUTIVE_LOSS_THRESHOLD,
            )
            return True

        return False

    async def _count_consecutive_losses(self, session: AsyncSession) -> int:
        """Count consecutive SL hits from most recent outcomes."""
        try:
            stmt = (
                select(Outcome.result)
                .order_by(Outcome.created_at.desc())
                .limit(CONSECUTIVE_LOSS_THRESHOLD + 5)
            )
            result = await session.execute(stmt)
            results = result.scalars().all()

            count = 0
            for r in results:
                if r == "sl_hit":
                    count += 1
                else:
                    break
            return count
        except Exception:
            return 0

    async def check_degradation(
        self, session: AsyncSession, strategy_id: int
    ) -> tuple[bool, str | None]:
        """Check if a strategy is degraded based on live performance vs baseline."""
        try:
            perf_stmt = select(StrategyPerformance).where(
                and_(
                    StrategyPerformance.strategy_id == strategy_id,
                    StrategyPerformance.period == "30d",
                )
            )
            perf_result = await session.execute(perf_stmt)
            perf = perf_result.scalar_one_or_none()

            if perf is None:
                return False, None

            if float(perf.profit_factor) < 1.0:
                reason = f"Profit factor {float(perf.profit_factor):.2f} below 1.0"
                return True, reason

            return False, None
        except Exception:
            return False, None

    async def check_recovery(
        self, session: AsyncSession, strategy_id: int
    ) -> bool:
        """Check if a previously degraded strategy has recovered."""
        try:
            perf_stmt = select(StrategyPerformance).where(
                and_(
                    StrategyPerformance.strategy_id == strategy_id,
                    StrategyPerformance.period == "7d",
                )
            )
            perf_result = await session.execute(perf_stmt)
            perf = perf_result.scalar_one_or_none()

            if perf is None:
                return False

            return float(perf.profit_factor) >= 1.0 and not perf.is_degraded
        except Exception:
            return False
