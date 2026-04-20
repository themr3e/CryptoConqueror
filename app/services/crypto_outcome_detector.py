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

# High water mark per signal for trailing stop (signal_id → best price seen)
_signal_high_water: dict[int, float] = {}

# ROI table: minutes open → minimum profit ratio to exit
# Exit at breakeven after 40m, take 1% after 30m, 2% after 20m
_ROI_TABLE: dict[int, float] = {40: 0.0, 30: 0.01, 20: 0.02}


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

        # ROI table: time-based exit — runs before SL/TP so time exits aren't blocked
        if self._check_roi_table(signal, price, now):
            return await self._record_outcome(session, signal, "tp1_hit", price, now)

        # Trail SL using Freqtrade positive offset pattern.
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

    @staticmethod
    def _check_roi_table(signal: Signal, price: Decimal, now: datetime) -> bool:
        """Return True if the ROI table says to exit this trade now."""
        if not signal.created_at:
            return False
        created = signal.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_minutes = (now - created).total_seconds() / 60

        entry = float(signal.entry_price)
        if entry <= 0:
            return False

        current = float(price)
        if signal.direction == "BUY":
            profit_ratio = (current - entry) / entry
        else:
            profit_ratio = (entry - current) / entry

        # Walk from longest open time to shortest — use first threshold that applies
        for min_minutes in sorted(_ROI_TABLE.keys(), reverse=True):
            if age_minutes >= min_minutes:
                return profit_ratio >= _ROI_TABLE[min_minutes]

        return False

    async def _maybe_trail_stop(
        self,
        session: AsyncSession,
        signal: Signal,
        current_price: Decimal,
    ) -> None:
        """Freqtrade positive-offset trailing stop.

        Activate: price moves +3% from entry in our favour.
        Trail:    SL moves to 2% below (BUY) or above (SELL) the highest
                  favourable price seen since activation.
        SL only tightens — never loosens.
        """
        from app.config import get_settings
        settings = get_settings()
        if not settings.binance_order_execution_enabled:
            return

        entry = float(signal.entry_price)
        sl    = float(signal.stop_loss)
        price = float(current_price)

        if signal.direction == "BUY":
            profit_pct = (price - entry) / entry if entry > 0 else 0.0
            if profit_pct < 0.03:
                return  # wait for +3% before engaging

            # Track high water mark
            hw = _signal_high_water.get(signal.id, price)
            hw = max(hw, price)
            _signal_high_water[signal.id] = hw

            new_sl = hw * (1.0 - 0.02)  # trail 2% below high water
            if new_sl <= sl:
                return  # no improvement
            if new_sl >= price:
                return  # would instantly close

        else:  # SELL
            profit_pct = (entry - price) / entry if entry > 0 else 0.0
            if profit_pct < 0.03:
                return

            # Track low water mark
            hw = _signal_high_water.get(signal.id, price)
            hw = min(hw, price)
            _signal_high_water[signal.id] = hw

            new_sl = hw * (1.0 + 0.02)  # trail 2% above low water
            if new_sl >= sl:
                return
            if new_sl <= price:
                return

        try:
            from app.services.binance_executor import BinanceExecutor
            executor = BinanceExecutor()
            updated = await executor.update_stop_loss(
                session, signal, Decimal(str(round(new_sl, 8)))
            )
            if updated:
                logger.info(
                    "CryptoOutcomeDetector: trail SL {} {} {:.4f} → {:.4f} (profit={:.1%})",
                    signal.symbol, signal.direction, sl, new_sl, profit_pct,
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
        pos_size = signal.position_size if signal.position_size else Decimal("1")
        raw_pnl = (exit_price - signal.entry_price) * direction_mult * pos_size

        taker_fee = await self._fee_model.get_taker_fee(session, signal.symbol)
        fee_cost = signal.entry_price * taker_fee * Decimal("2") * pos_size
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

        # Clean up high water mark for this signal
        _signal_high_water.pop(signal.id, None)

        # Enforce 2h cooldown on this symbol after any exit
        try:
            from app.services.risk_manager import engage_cooldown
            engage_cooldown(signal.symbol)
        except Exception:
            pass

        # Write a micro-lesson after every stop-loss hit
        if result == "sl_hit":
            try:
                from app.services.self_improver import SelfImprover
                await SelfImprover().analyze_loss(session, signal, outcome)
            except Exception:
                logger.opt(exception=True).warning("SelfImprover: loss lesson failed")

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
