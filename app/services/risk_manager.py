"""Risk manager service: capital protection and position sizing.

Enforces multiple risk constraints before approving trade signals:
circuit breaker, daily loss limit, concurrent signal cap, and
volatility-adjusted position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from loguru import logger
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.outcome import Outcome
from app.models.signal import Signal

RISK_PER_TRADE = 0.01       # 1% of account balance
MAX_CONCURRENT_SIGNALS = 3
DAILY_LOSS_LIMIT = 0.02     # 2% account drawdown
ATR_FACTOR_MIN = 0.5
ATR_FACTOR_MAX = 1.5


@dataclass
class RiskCheckResult:
    """Result of a risk check for a single candidate signal."""
    approved: bool
    rejection_reason: str | None
    position_size: Decimal
    risk_amount: float
    daily_pnl: float


class RiskManager:
    """Validates candidate signals against risk constraints."""

    async def check(
        self,
        session: AsyncSession,
        candidates: list,
        current_atr: float = 1.0,
        baseline_atr: float = 1.0,
    ) -> list[tuple]:
        """Run all risk checks on candidate signals.

        Returns:
            List of (candidate, RiskCheckResult) tuples.
        """
        settings = get_settings()
        account_balance = settings.account_balance

        # 1. Circuit breaker check
        cb_active = await self._check_circuit_breaker(session)
        if cb_active:
            return [
                (c, RiskCheckResult(
                    approved=False,
                    rejection_reason="Circuit breaker active",
                    position_size=Decimal("0"),
                    risk_amount=0.0,
                    daily_pnl=0.0,
                ))
                for c in candidates
            ]

        # 2. Daily loss limit check (daily_pnl is in USDT for crypto)
        daily_pnl = await self._check_daily_loss(session)
        daily_loss_pct = abs(daily_pnl / account_balance) if daily_pnl < 0 else 0.0
        if daily_loss_pct >= DAILY_LOSS_LIMIT:
            return [
                (c, RiskCheckResult(
                    approved=False,
                    rejection_reason=f"Daily loss limit reached ({daily_loss_pct:.1%})",
                    position_size=Decimal("0"),
                    risk_amount=0.0,
                    daily_pnl=daily_pnl,
                ))
                for c in candidates
            ]

        # 3. Concurrent signal check
        concurrent_count = await self._check_concurrent_limit(session)
        if concurrent_count >= MAX_CONCURRENT_SIGNALS:
            return [
                (c, RiskCheckResult(
                    approved=False,
                    rejection_reason=f"Max concurrent signals reached ({concurrent_count}/{MAX_CONCURRENT_SIGNALS})",
                    position_size=Decimal("0"),
                    risk_amount=0.0,
                    daily_pnl=daily_pnl,
                ))
                for c in candidates
            ]

        # 4. Position sizing and individual approval
        results = []
        for candidate in candidates:
            position_size, risk_amount = self.calculate_position_size(
                candidate, account_balance, current_atr, baseline_atr
            )

            results.append((
                candidate,
                RiskCheckResult(
                    approved=True,
                    rejection_reason=None,
                    position_size=position_size,
                    risk_amount=risk_amount,
                    daily_pnl=daily_pnl,
                ),
            ))

        return results

    async def _check_circuit_breaker(self, session: AsyncSession) -> bool:
        """Check if circuit breaker is currently active."""
        try:
            from app.services.feedback_controller import FeedbackController
            fb = FeedbackController()
            return await fb.check_circuit_breaker(session)
        except Exception:
            return False

    async def _check_daily_loss(self, session: AsyncSession) -> float:
        """Sum today's P&L in USDT."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        try:
            stmt = select(func.coalesce(func.sum(Outcome.pnl_usdt), 0)).where(
                Outcome.created_at >= today_start
            )
            result = await session.execute(stmt)
            return float(result.scalar_one())
        except Exception:
            return 0.0

    async def _check_concurrent_limit(self, session: AsyncSession) -> int:
        """Count currently active signals."""
        try:
            stmt = select(func.count()).select_from(Signal).where(Signal.status == "active")
            result = await session.execute(stmt)
            return result.scalar_one()
        except Exception:
            return 0

    def calculate_position_size(
        self,
        candidate,
        account_balance: float,
        current_atr: float,
        baseline_atr: float,
    ) -> tuple[Decimal, float]:
        """Calculate volatility-adjusted position size.

        Returns:
            (position_size, risk_amount)
        """
        risk_amount = account_balance * RISK_PER_TRADE

        sl_distance = abs(float(candidate.entry_price) - float(candidate.stop_loss))
        if sl_distance <= 0:
            sl_distance = 0.1  # Fallback to prevent division by zero

        # ATR adjustment factor: reduce size in high-vol, increase in low-vol
        if baseline_atr > 0 and current_atr > 0:
            atr_factor = baseline_atr / current_atr
            atr_factor = max(ATR_FACTOR_MIN, min(ATR_FACTOR_MAX, atr_factor))
        else:
            atr_factor = 1.0

        raw_size = (risk_amount / sl_distance) * atr_factor
        position_size = Decimal(str(round(raw_size, 2)))

        logger.debug(
            "Position sizing: risk={:.2f} sl_dist={:.4f} atr_factor={:.3f} size={}",
            risk_amount,
            sl_distance,
            atr_factor,
            position_size,
        )

        return position_size, risk_amount

    async def get_drawdown_metrics(self, session: AsyncSession) -> dict:
        """Compute running and maximum drawdown from historical outcomes."""
        try:
            stmt = select(Outcome.pnl_usdt).order_by(Outcome.created_at.asc())
            result = await session.execute(stmt)
            pnl_values = [float(r) for r in result.scalars().all()]

            if not pnl_values:
                return {"current_drawdown": 0.0, "max_drawdown": 0.0}

            cumulative = 0.0
            peak = 0.0
            max_dd = 0.0

            for pnl in pnl_values:
                cumulative += pnl
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd

            current_dd = max(0.0, peak - cumulative)
            return {"current_drawdown": current_dd, "max_drawdown": max_dd}
        except Exception:
            return {"current_drawdown": 0.0, "max_drawdown": 0.0}
