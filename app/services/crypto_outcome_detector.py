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
_CACHE_TTL_SECONDS = 30  # 30 seconds — must be shorter than the 2-min detection interval


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

        # Trail SL toward entry (or beyond) when sufficiently in profit.
        # Must run BEFORE the SL/TP check so the tighter stop takes effect
        # immediately in the same evaluation cycle.
        await self._maybe_trail_stop(session, signal, price)

        # Re-read stop_loss from the signal object — _maybe_trail_stop may
        # have updated it in-memory via the ORM.
        entry = signal.entry_price
        sl    = signal.stop_loss
        tp1   = signal.take_profit_1
        tp2   = signal.take_profit_2

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

    async def _maybe_trail_stop(
        self,
        session: AsyncSession,
        signal: Signal,
        current_price: Decimal,
    ) -> None:
        """Move SL toward or past entry when the trade is sufficiently in profit.

        Two stages:
          Stage 1 — Break-even  (profit ≥ 50 % of initial risk):
              Move SL to entry price.  Worst case from here is 0 loss.
          Stage 2 — Trailing    (profit ≥ 100 % of initial risk):
              Trail SL at 50 % of initial-risk distance behind current price.
              This locks in a portion of the profit as the trade extends.

        A minimum improvement threshold (5 % of initial risk) prevents
        excessive order churn on micro price wiggles.
        """
        from app.config import get_settings
        settings = get_settings()
        if not settings.binance_order_execution_enabled:
            return  # no API keys — skip Binance order management

        entry = float(signal.entry_price)
        sl    = float(signal.stop_loss)
        price = float(current_price)

        initial_risk = abs(entry - sl)
        if initial_risk <= 0:
            return

        new_sl: float

        if signal.direction == "BUY":
            profit_distance = price - entry
            if profit_distance <= 0:
                return  # not in profit yet
            profit_ratio = profit_distance / initial_risk

            if profit_ratio < 0.5:
                return  # not enough profit to start trailing

            # Stage 1: break-even
            new_sl = entry
            # Stage 2: trail 0.5 × initial_risk behind price
            if profit_ratio >= 1.0:
                new_sl = max(new_sl, price - initial_risk * 0.5)

            # Only update if the improvement is meaningful
            if new_sl - sl < initial_risk * 0.05:
                return
            # SL must always stay below price for BUY (otherwise we'd instant-close)
            if new_sl >= price:
                return

        else:  # SELL — profit = price falling below entry
            profit_distance = entry - price
            if profit_distance <= 0:
                return
            profit_ratio = profit_distance / initial_risk

            if profit_ratio < 0.5:
                return

            # Stage 1: break-even
            new_sl = entry
            # Stage 2: trail 0.5 × initial_risk above price
            if profit_ratio >= 1.0:
                new_sl = min(new_sl, price + initial_risk * 0.5)

            if sl - new_sl < initial_risk * 0.05:
                return
            # SL must always stay above price for SELL
            if new_sl <= price:
                return

        # Apply the trailing update via BinanceExecutor
        try:
            from app.services.binance_executor import BinanceExecutor
            executor = BinanceExecutor()
            updated = await executor.update_stop_loss(
                session, signal, Decimal(str(round(new_sl, 8)))
            )
            if updated:
                logger.info(
                    "CryptoOutcomeDetector: trailed SL {} {} {:.4f} → {:.4f} "
                    "(profit_ratio={:.2f}x)",
                    signal.symbol, signal.direction, sl, new_sl, profit_ratio,
                )
        except Exception:
            logger.opt(exception=True).warning(
                "CryptoOutcomeDetector: trail SL failed for {}", signal.symbol
            )

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
