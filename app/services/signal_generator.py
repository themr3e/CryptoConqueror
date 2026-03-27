"""Signal generator service: generation, validation, dedup, expiry, and bias detection.

Transforms strategy analysis into validated, de-duplicated trade signals.

Float math internally; Decimal(str(round(x, 2))) at persistence boundary only.
"""

from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.candle import Candle
from app.models.signal import Signal

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MIN_RR: float = 1.3
MIN_CONFIDENCE: float = 40.0
MAX_SL_PIPS: float = 800.0
PIP_VALUE: float = 0.10
DEDUP_WINDOW_HOURS: int = 1
EXPIRY_HOURS: dict[str, int] = {
    "M15": 4,
    "H1": 8,
    "H4": 24,
    "D1": 48,
}
BIAS_WINDOW_SIGNALS: int = 20
BIAS_SKEW_THRESHOLD: float = 0.75


class SignalGenerator:
    """Generates, validates, deduplicates, and expires trade signals."""

    async def generate(
        self,
        session: AsyncSession,
        strategy_name: str,
        symbol: str = "BTCUSDT",
    ) -> list:
        """Run a strategy's generate_signals() on latest candle data.

        Args:
            session:       Async DB session.
            strategy_name: Registered strategy name.
            symbol:        Market symbol to load candles for (e.g. ``"BTCUSDT"``, ``"ETHUSDT"``).

        Returns:
            List of CandidateSignal instances.
        """
        import pandas as pd
        import app.strategies  # noqa: F401 — triggers all self-registrations
        from app.strategies.base import BaseStrategy

        opt_params = await self._load_optimized_params(session, strategy_name)

        registry = BaseStrategy.get_registry()
        if strategy_name not in registry:
            logger.error(
                "Strategy '{}' not found in registry. Available: {}",
                strategy_name,
                list(registry.keys()),
            )
            return []

        strategy_cls = registry[strategy_name]
        strategy = strategy_cls(params=opt_params)

        # Load H1 candles (primary timeframe for all strategies)
        limit = 350  # sufficient for all strategy lookbacks + EMA200
        stmt = (
            select(Candle)
            .where(and_(Candle.symbol == symbol, Candle.timeframe == "H1"))
            .order_by(Candle.timestamp.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        candles_orm = result.scalars().all()

        if not candles_orm:
            logger.warning("No candles found for {}/H1 -- cannot generate signals", symbol)
            return []

        # Build DataFrame (oldest first)
        rows = list(reversed(candles_orm))
        df = pd.DataFrame([{
            "timestamp": c.timestamp,
            "open":   float(c.open),
            "high":   float(c.high),
            "low":    float(c.low),
            "close":  float(c.close),
            "volume": float(c.volume) if c.volume is not None else 0.0,
        } for c in rows]).set_index("timestamp")
        df.attrs["symbol"] = symbol  # let strategies infer the symbol

        try:
            from app.strategies.base import CandidateSignal
            candidates: list[CandidateSignal] = strategy.generate_signals(df)
        except Exception as exc:
            logger.opt(exception=True).warning(
                "Strategy '{}' raised an exception: {}", strategy_name, str(exc)[:200]
            )
            return []

        if candidates:
            tf_hours = {"M15": 0.25, "H1": 1, "H4": 4, "D1": 24}
            interval_hours = tf_hours.get("H1", 1)  # candles are always loaded as H1
            staleness_cutoff = datetime.now(timezone.utc) - timedelta(
                hours=interval_hours * 3
            )

            fresh = []
            stale_count = 0
            for c in candidates:
                c_ts = c.timestamp
                if c_ts is not None:
                    if c_ts.tzinfo is None:
                        c_ts = c_ts.replace(tzinfo=timezone.utc)
                    if c_ts >= staleness_cutoff:
                        fresh.append(c)
                    else:
                        stale_count += 1
                else:
                    fresh.append(c)

            if stale_count > 0:
                logger.info(
                    "Strategy '{}': filtered {} stale candidates, {} fresh remain",
                    strategy_name,
                    stale_count,
                    len(fresh),
                )
            candidates = fresh

        if candidates:
            logger.info(
                "Strategy '{}' produced {} candidate signal(s)",
                strategy_name,
                len(candidates),
            )
        else:
            logger.info(
                "Strategy '{}' produced 0 candidates from {} candles",
                strategy_name,
                len(df),
            )
        return candidates

    @staticmethod
    async def _load_optimized_params(
        session: AsyncSession,
        strategy_name: str,
    ) -> dict[str, float] | None:
        """Load active, non-overfitted optimized params for a strategy."""
        try:
            from app.models.optimized_params import OptimizedParams

            stmt = (
                select(OptimizedParams.params)
                .where(
                    and_(
                        OptimizedParams.strategy_name == strategy_name,
                        OptimizedParams.is_active.is_(True),
                        OptimizedParams.is_overfitted.isnot(True),
                    )
                )
                .order_by(OptimizedParams.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return row if row else None
        except Exception as exc:
            logger.warning(
                "Could not load optimized params for '{}': {} -- using defaults",
                strategy_name,
                str(exc)[:120],
            )
            await session.rollback()
            return None

    async def validate(
        self,
        session: AsyncSession,
        candidates: list,
    ) -> list:
        """Apply validation filters to candidate signals."""
        validated: list = []

        for candidate in candidates:
            rr = float(candidate.risk_reward)
            if rr < MIN_RR:
                logger.info(
                    "Signal rejected: R:R {:.2f} below minimum {:.2f}",
                    rr, MIN_RR,
                )
                continue

            sl_dist = abs(float(candidate.entry_price) - float(candidate.stop_loss))
            entry = float(candidate.entry_price)

            # Percentage-based SL check (max 5% from entry)
            sl_pct = (sl_dist / entry * 100) if entry > 0 else 0
            if sl_pct > 5.0:
                logger.info(
                    "Signal rejected: SL {:.2f}% from entry exceeds 5%",
                    sl_pct,
                )
                continue

            conf = float(candidate.confidence)
            if conf < MIN_CONFIDENCE:
                logger.info(
                    "Signal rejected: confidence {:.1f}% below minimum {:.1f}%",
                    conf, MIN_CONFIDENCE,
                )
                continue

            if await self._is_duplicate(session, candidate):
                logger.info(
                    "Signal suppressed: duplicate {} signal within {}h window",
                    candidate.direction.value,
                    DEDUP_WINDOW_HOURS,
                )
                continue

            if await self._check_directional_bias(session, candidate):
                logger.warning(
                    "Directional bias detected: >{}% of recent signals are {}",
                    int(BIAS_SKEW_THRESHOLD * 100),
                    candidate.direction.value,
                )
                candidate = candidate.model_copy(
                    update={
                        "reasoning": (
                            candidate.reasoning
                            + f" [NOTE: directional bias detected"
                            f" -- >{int(BIAS_SKEW_THRESHOLD * 100)}% of"
                            f" last {BIAS_WINDOW_SIGNALS} signals are"
                            f" {candidate.direction.value}]"
                        ),
                    }
                )

            validated.append(candidate)

        logger.info(
            "Validation complete: {}/{} candidates passed all filters",
            len(validated),
            len(candidates),
        )
        return validated

    async def _is_duplicate(self, session: AsyncSession, candidate: object) -> bool:
        """Check if an active signal with the same direction exists within the dedup window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=DEDUP_WINDOW_HOURS)

        stmt = (
            select(Signal.id)
            .where(
                and_(
                    Signal.symbol == candidate.symbol,
                    Signal.direction == candidate.direction.value,
                    Signal.status == "active",
                    Signal.created_at >= cutoff,
                )
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def _check_directional_bias(
        self, session: AsyncSession, candidate: object
    ) -> bool:
        """Detect if recent signal distribution is systematically skewed."""
        stmt = (
            select(Signal.direction)
            .order_by(Signal.created_at.desc())
            .limit(BIAS_WINDOW_SIGNALS)
        )
        result = await session.execute(stmt)
        directions = result.scalars().all()

        if len(directions) < BIAS_WINDOW_SIGNALS:
            return False

        same_direction_count = sum(
            1 for d in directions if d == candidate.direction.value
        )
        ratio = same_direction_count / len(directions)

        return ratio > BIAS_SKEW_THRESHOLD

    def compute_expiry(self, candidate: object) -> datetime:
        """Compute the expiry timestamp for a candidate signal."""
        expiry_hours = EXPIRY_HOURS.get(candidate.timeframe, 8)
        return datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

    async def expire_stale_signals(self, session: AsyncSession) -> int:
        """Mark active signals past their expiry as expired."""
        now = datetime.now(timezone.utc)

        stmt = (
            update(Signal)
            .where(
                and_(
                    Signal.status == "active",
                    Signal.expires_at.isnot(None),
                    Signal.expires_at < now,
                )
            )
            .values(status="expired")
        )
        result = await session.execute(stmt)
        count = result.rowcount

        if count > 0:
            logger.info("Expired {} stale signal(s)", count)
        else:
            logger.debug("No stale signals to expire")

        return count
