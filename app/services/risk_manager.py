"""Risk manager service: capital protection and position sizing.

Enforces multiple risk constraints before approving trade signals:
circuit breaker, daily loss limit, concurrent signal cap,
volatility-adjusted position sizing, and Freqtrade-style protections
(StoplossGuard, MaxDrawdown, CooldownPeriod, LowProfitPairs).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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

# H1 candle = 60 minutes
_CANDLE_MINUTES = 60

# Global locks — separate so MaxDrawdown can extend beyond StoplossGuard
_stoploss_guard_lock: datetime | None = None
_max_drawdown_lock: datetime | None = None

# Per-symbol lock: set by CooldownPeriod and LowProfitPairs
_symbol_lock_until: dict[str, datetime] = {}


def engage_cooldown(symbol: str) -> None:
    """Enforce 2-candle (2h) cooldown after any trade exit on a symbol."""
    global _symbol_lock_until
    lock_until = datetime.now(timezone.utc) + timedelta(minutes=_CANDLE_MINUTES * 2)
    existing = _symbol_lock_until.get(symbol)
    if existing is None or lock_until > existing:
        _symbol_lock_until[symbol] = lock_until
        logger.info("RiskManager: CooldownPeriod engaged for {} until {}", symbol, lock_until.strftime("%H:%M UTC"))


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

        # 3. StoplossGuard: halt after 4 SL hits in 24h → lock 4h
        sl_blocked, sl_reason = await self._check_stoploss_guard(session)
        if sl_blocked:
            return [
                (c, RiskCheckResult(
                    approved=False,
                    rejection_reason=sl_reason,
                    position_size=Decimal("0"),
                    risk_amount=0.0,
                    daily_pnl=daily_pnl,
                ))
                for c in candidates
            ]

        # 4. MaxDrawdown: halt if equity drops 20% in 48h → lock 12h
        dd_blocked, dd_reason = await self._check_max_drawdown_guard(session)
        if dd_blocked:
            return [
                (c, RiskCheckResult(
                    approved=False,
                    rejection_reason=dd_reason,
                    position_size=Decimal("0"),
                    risk_amount=0.0,
                    daily_pnl=daily_pnl,
                ))
                for c in candidates
            ]

        # 5. Concurrent signal check
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

        # 6. Per-candidate: CooldownPeriod + LowProfitPairs + position sizing
        results = []
        for candidate in candidates:
            symbol = getattr(candidate, "symbol", "")

            # CooldownPeriod: in-memory, 2h after any exit
            cooldown_blocked, cooldown_reason = self._check_cooldown(symbol)
            if cooldown_blocked:
                results.append((candidate, RiskCheckResult(
                    approved=False,
                    rejection_reason=cooldown_reason,
                    position_size=Decimal("0"),
                    risk_amount=0.0,
                    daily_pnl=daily_pnl,
                )))
                continue

            # LowProfitPairs: lock symbol 60 min if <2% profit in last 6h (min 2 trades)
            lp_blocked, lp_reason = await self._check_low_profit_pair(session, symbol)
            if lp_blocked:
                results.append((candidate, RiskCheckResult(
                    approved=False,
                    rejection_reason=lp_reason,
                    position_size=Decimal("0"),
                    risk_amount=0.0,
                    daily_pnl=daily_pnl,
                )))
                continue

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

    async def _check_stoploss_guard(self, session: AsyncSession) -> tuple[bool, str]:
        """Halt if 4+ SL hits in last 24 candles (24h). Lock 4 candles (4h)."""
        global _stoploss_guard_lock
        now = datetime.now(timezone.utc)
        if _stoploss_guard_lock and now < _stoploss_guard_lock:
            return True, f"StoplossGuard: locked until {_stoploss_guard_lock.strftime('%H:%M UTC')}"
        try:
            cutoff = now - timedelta(hours=24)
            stmt = select(func.count()).select_from(Outcome).where(
                and_(Outcome.created_at >= cutoff, Outcome.result == "sl_hit")
            )
            result = await session.execute(stmt)
            sl_count = int(result.scalar_one() or 0)
        except Exception:
            return False, ""
        if sl_count >= 4:
            _stoploss_guard_lock = now + timedelta(hours=4)
            logger.warning(
                "RiskManager: StoplossGuard — {} SL hits in 24h, locked 4h until {}",
                sl_count, _stoploss_guard_lock.strftime("%H:%M UTC"),
            )
            return True, f"StoplossGuard: {sl_count} SL hits in 24h — locked 4h"
        return False, ""

    async def _check_max_drawdown_guard(self, session: AsyncSession) -> tuple[bool, str]:
        """Halt if equity drops 20% in last 48 candles (48h). Lock 12 candles (12h)."""
        global _max_drawdown_lock
        now = datetime.now(timezone.utc)
        if _max_drawdown_lock and now < _max_drawdown_lock:
            return True, f"MaxDrawdown: locked until {_max_drawdown_lock.strftime('%H:%M UTC')}"
        try:
            settings = get_settings()
            cutoff = now - timedelta(hours=48)
            stmt = select(func.coalesce(func.sum(Outcome.pnl_usdt), 0)).where(
                Outcome.created_at >= cutoff
            )
            result = await session.execute(stmt)
            period_pnl = float(result.scalar_one())
        except Exception:
            return False, ""
        drawdown_pct = abs(period_pnl) / settings.account_balance if period_pnl < 0 else 0.0
        if drawdown_pct >= 0.20:
            _max_drawdown_lock = now + timedelta(hours=12)
            logger.warning(
                "RiskManager: MaxDrawdown — {:.1%} equity drop in 48h, locked 12h until {}",
                drawdown_pct, _max_drawdown_lock.strftime("%H:%M UTC"),
            )
            return True, f"MaxDrawdown: {drawdown_pct:.1%} in 48h — locked 12h"
        return False, ""

    def _check_cooldown(self, symbol: str) -> tuple[bool, str]:
        """CooldownPeriod: block symbol for 2h after any exit (in-memory)."""
        if not symbol:
            return False, ""
        now = datetime.now(timezone.utc)
        lock_until = _symbol_lock_until.get(symbol)
        if lock_until and now < lock_until:
            return True, f"CooldownPeriod: {symbol} locked until {lock_until.strftime('%H:%M UTC')}"
        return False, ""

    async def _check_low_profit_pair(self, session: AsyncSession, symbol: str) -> tuple[bool, str]:
        """LowProfitPairs: lock symbol 60 min if net loss in last 6h with ≥2 trades."""
        if not symbol:
            return False, ""
        now = datetime.now(timezone.utc)
        lock_until = _symbol_lock_until.get(symbol)
        if lock_until and now < lock_until:
            return True, f"LowProfitPairs: {symbol} locked until {lock_until.strftime('%H:%M UTC')}"
        try:
            cutoff = now - timedelta(hours=6)
            stmt = (
                select(Outcome.pnl_usdt)
                .join(Signal, Outcome.signal_id == Signal.id)
                .where(and_(Signal.symbol == symbol, Outcome.created_at >= cutoff))
            )
            result = await session.execute(stmt)
            pnl_values = [float(r) for r in result.scalars().all()]
        except Exception:
            return False, ""
        if len(pnl_values) < 2:
            return False, ""
        total_pnl = sum(pnl_values)
        if total_pnl < 0:
            new_lock = now + timedelta(minutes=60)
            _symbol_lock_until[symbol] = new_lock
            logger.warning(
                "RiskManager: LowProfitPairs — {} net ${:.2f} in 6h ({} trades), locked 60m",
                symbol, total_pnl, len(pnl_values),
            )
            return True, f"LowProfitPairs: {symbol} net ${total_pnl:.2f} in 6h — locked 60m"
        return False, ""

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
