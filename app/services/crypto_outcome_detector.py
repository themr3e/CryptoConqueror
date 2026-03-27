"""Crypto futures outcome detector.

Parallel to ``OutcomeDetector`` but uses Binance mark price instead of
Twelve Data for price fetching, and records P&L in USDT via ``pnl_usdt``.

Only processes signals where ``signal.symbol IN crypto_symbols``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outcome import Outcome
from app.models.signal import Signal
from app.services.crypto_candle_ingestor import CryptoCandleIngestor
from app.services.crypto_fee_model import CryptoFeeModel
from app.services.performance_tracker import PerformanceTracker
from app.services.telegram_notifier import TelegramNotifier

# Price cache: symbol → (price, fetched_at)
_price_cache: dict[str, tuple[float, datetime]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


class CryptoOutcomeDetector:
    """Checks active crypto signals against Binance mark price and records outcomes."""

    def __init__(
        self,
        crypto_ingestor: CryptoCandleIngestor,
        fee_model: CryptoFeeModel,
        notifier: TelegramNotifier | None = None,
        perf_tracker: PerformanceTracker | None = None,
    ) -> None:
        self._ingestor = crypto_ingestor
        self._fee_model = fee_model
        self._notifier = notifier or TelegramNotifier()
        self._perf_tracker = perf_tracker or PerformanceTracker()

    async def check_active_signals(
        self,
        session: AsyncSession,
        crypto_symbols: list[str] | None = None,
    ) -> list[Outcome]:
        """Check all active crypto signals and record outcomes if hit.

        Args:
            session:        Async DB session.
            crypto_symbols: List of symbols to check (e.g. ``["BTCUSDT", "ETHUSDT"]``).
                            If None, fetches all active signals.

        Returns:
            List of newly created Outcome records.
        """
        stmt = select(Signal).where(Signal.status == "active")
        if crypto_symbols:
            stmt = stmt.where(Signal.symbol.in_(crypto_symbols))

        result = await session.execute(stmt)
        signals = result.scalars().all()

        if not signals:
            return []

        outcomes: list[Outcome] = []
        for signal in signals:
            outcome = await self._evaluate_signal(session, signal)
            if outcome:
                outcomes.append(outcome)

        return outcomes

    async def _evaluate_signal(
        self,
        session: AsyncSession,
        signal: Signal,
    ) -> Outcome | None:
        """Evaluate a single signal against current mark price."""
        mark_price = await self._get_mark_price_cached(signal.symbol)
        if mark_price is None:
            logger.warning(
                "CryptoOutcomeDetector: could not fetch mark price for {}", signal.symbol
            )
            return None

        now = datetime.now(timezone.utc)

        # Check expiry first
        if signal.expires_at and now >= signal.expires_at:
            return await self._record_outcome(
                session, signal, "expired", Decimal(str(mark_price)), now
            )

        price = Decimal(str(mark_price))
        entry = signal.entry_price
        sl = signal.stop_loss
        tp1 = signal.take_profit_1
        tp2 = signal.take_profit_2

        if signal.direction == "BUY":
            if price <= sl:
                return await self._record_outcome(session, signal, "sl_hit", price, now)
            if tp2 and price >= tp2:
                return await self._record_outcome(session, signal, "tp2_hit", price, now)
            if price >= tp1:
                return await self._record_outcome(session, signal, "tp1_hit", price, now)
        else:  # SELL
            if price >= sl:
                return await self._record_outcome(session, signal, "sl_hit", price, now)
            if tp2 and price <= tp2:
                return await self._record_outcome(session, signal, "tp2_hit", price, now)
            if price <= tp1:
                return await self._record_outcome(session, signal, "tp1_hit", price, now)

        return None

    async def _record_outcome(
        self,
        session: AsyncSession,
        signal: Signal,
        result: str,
        exit_price: Decimal,
        now: datetime,
    ) -> Outcome:
        """Persist outcome and update signal status."""
        # Calculate P&L in USDT
        direction_mult = Decimal("1") if signal.direction == "BUY" else Decimal("-1")
        raw_pnl = (exit_price - signal.entry_price) * direction_mult

        # Subtract round-trip taker fee (approximate position size = 1 contract)
        taker_fee = await self._fee_model.get_taker_fee(session, signal.symbol)
        fee_cost = signal.entry_price * taker_fee * 2
        pnl_usdt = raw_pnl - fee_cost

        duration_minutes: int | None = None
        if signal.created_at:
            created_aware = signal.created_at
            if created_aware.tzinfo is None:
                created_aware = created_aware.replace(tzinfo=timezone.utc)
            duration_minutes = int((now - created_aware).total_seconds() / 60)

        outcome = Outcome(
            signal_id=signal.id,
            result=result,
            exit_price=exit_price,
            pnl_pips=Decimal("0"),       # N/A for crypto; use pnl_usdt
            pnl_usdt=pnl_usdt,
            duration_minutes=duration_minutes,
        )
        session.add(outcome)

        signal.status = "closed"
        await session.commit()

        logger.info(
            "CryptoOutcomeDetector: {} {} {} @ {} → pnl_usdt={}",
            signal.symbol, signal.direction, result, exit_price, round(pnl_usdt, 4),
        )

        # Notify
        try:
            await self._notifier.notify_outcome(signal, outcome)
        except Exception:
            logger.opt(exception=True).warning("Failed to send outcome Telegram notification")

        # Update performance tracker
        try:
            await self._perf_tracker.record_outcome(session, signal, outcome)
        except Exception:
            logger.opt(exception=True).warning("Failed to update performance after crypto outcome")

        return outcome

    async def _get_mark_price_cached(self, symbol: str) -> float | None:
        """Return cached mark price or fetch fresh if stale."""
        now = datetime.now(timezone.utc)
        cached = _price_cache.get(symbol)
        if cached is not None:
            price, fetched_at = cached
            if (now - fetched_at).total_seconds() < _CACHE_TTL_SECONDS:
                return price

        price = await self._ingestor.get_mark_price(symbol)
        if price is not None:
            _price_cache[symbol] = (price, now)
        return price
