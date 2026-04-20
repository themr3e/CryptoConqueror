"""Signal pipeline orchestrator: the heartbeat of the trading system.

Wires StrategySelector, SignalGenerator, and RiskManager into a sequential
flow that runs every hour for crypto futures symbols.

Exports:
    SignalPipeline -- main orchestrator class
"""

from __future__ import annotations

from decimal import Decimal

from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.candle import Candle
from app.models.signal import Signal
from app.models.strategy import Strategy as StrategyModel
from app.services.risk_manager import RiskManager
from app.services.signal_generator import SignalGenerator
from app.services.strategy_selector import StrategySelector
from app.strategies.helpers.indicators import compute_atr


class SignalPipeline:
    """Orchestrates the full signal generation pipeline."""

    def __init__(
        self,
        selector: StrategySelector,
        generator: SignalGenerator,
        risk_manager: RiskManager,
    ) -> None:
        self.selector = selector
        self.generator = generator
        self.risk_manager = risk_manager

    async def run(
        self,
        session: AsyncSession,
        symbol: str = "BTCUSDT",
    ) -> list[Signal]:
        """Execute the full signal pipeline for the given symbol.

        Args:
            session: Async DB session.
            symbol:  Crypto futures symbol to generate signals for.
        """
        asset_class = "crypto_futures"

        # 1. Expire stale signals
        expired_count = await self.generator.expire_stale_signals(session)
        logger.info("Expired {} stale signal(s) before scan (symbol={})", expired_count, symbol)

        # 2. Rank all strategies for this asset class
        ranked = await self.selector.select_all_ranked(session, asset_class=asset_class)
        if not ranked:
            logger.warning("No qualifying strategy found, skipping signal generation.")
            return []

        regime = ranked[0].regime
        logger.info(
            "Signal pipeline: {} strategies qualified, regime={}",
            len(ranked), regime.value,
        )

        # 3. Try each strategy in ranked order
        validated = []
        strategy_name = None
        approved_sizes: dict[int, Decimal] = {}

        for score in ranked:
            strategy_name = score.strategy_name
            logger.info(
                "Trying strategy '{}' (score={:.4f}, degraded={})",
                strategy_name, score.composite_score, score.is_degraded,
            )

            candidates = await self.generator.generate(session, strategy_name, symbol=symbol)
            if not candidates:
                continue

            valid = await self.generator.validate(session, candidates)
            if not valid:
                continue

            valid.sort(key=lambda c: float(c.confidence), reverse=True)
            best_candidate = valid[0]
            validated = [best_candidate]

            # Block opposite-direction signal for this symbol only
            active_stmt = (
                select(Signal.direction)
                .where(and_(Signal.status == "active", Signal.symbol == symbol))
                .limit(1)
            )
            active_result = await session.execute(active_stmt)
            active_dir = active_result.scalar_one_or_none()
            if active_dir is not None:
                new_dir = validated[0].direction.value
                if new_dir != active_dir:
                    logger.info(
                        "Blocking {} signal: active {} signal already open",
                        new_dir, active_dir,
                    )
                    validated = []
                    continue

            # Risk check
            current_atr, baseline_atr = await self._compute_atr(session, symbol=symbol)
            risk_results = await self.risk_manager.check(
                session, validated,
                current_atr=current_atr,
                baseline_atr=baseline_atr,
            )

            approved_candidates = []
            for i, (candidate, risk_result) in enumerate(risk_results):
                if risk_result.approved:
                    approved_candidates.append(candidate)
                    approved_sizes[len(approved_candidates) - 1] = risk_result.position_size
                else:
                    logger.info("Candidate rejected: {}", risk_result.rejection_reason)

            if not approved_candidates:
                validated = []
                continue

            validated = approved_candidates
            logger.info("Strategy '{}' produced {} approved candidate(s)", strategy_name, len(validated))
            break
        else:
            logger.warning("All strategies tried, none produced valid signals.")
            return []

        if not validated or strategy_name is None:
            return []

        # 4. H4 confluence boost
        for i, candidate in enumerate(validated):
            has_confluence = await self.selector.check_h4_confluence(
                session, candidate.direction.value, symbol=symbol
            )
            if has_confluence:
                boosted = min(float(candidate.confidence) + 5, 100.0)
                new_confidence = Decimal(str(round(boosted, 2)))
                validated[i] = candidate.model_copy(update={
                    "confidence": new_confidence,
                    "reasoning": candidate.reasoning + " | H4 confluence confirmed",
                })

        enriched = validated

        # 7. Persist signals
        strat_stmt = select(StrategyModel).where(StrategyModel.name == strategy_name)
        strat_result = await session.execute(strat_stmt)
        strategy_row = strat_result.scalar_one_or_none()

        if strategy_row is None:
            logger.error("Strategy '{}' not found in strategies table", strategy_name)
            return []

        strategy_id = strategy_row.id

        persisted: list[Signal] = []
        for i, candidate in enumerate(enriched):
            expires_at = self.generator.compute_expiry(candidate)
            position_size = approved_sizes.get(i)

            signal = Signal(
                strategy_id=strategy_id,
                symbol=candidate.symbol,
                timeframe=candidate.timeframe,
                direction=candidate.direction.value,
                entry_price=candidate.entry_price,
                stop_loss=candidate.stop_loss,
                take_profit_1=candidate.take_profit_1,
                take_profit_2=candidate.take_profit_2,
                risk_reward=candidate.risk_reward,
                confidence=candidate.confidence,
                reasoning=candidate.reasoning,
                position_size=position_size,
                status="active",
                expires_at=expires_at,
            )
            session.add(signal)
            persisted.append(signal)

        await session.commit()

        logger.info(
            "Pipeline complete: {} signal(s) generated from '{}' (symbol={}, regime={})",
            len(persisted), strategy_name, symbol, regime.value,
        )
        return persisted

    async def _compute_atr(
        self,
        session: AsyncSession,
        symbol: str = "BTCUSDT",
    ) -> tuple[float, float]:
        """Compute current and baseline ATR(14) from H1 candle data for the given symbol."""
        import pandas as pd

        stmt = (
            select(Candle.high, Candle.low, Candle.close)
            .where(
                and_(
                    Candle.symbol == symbol,
                    Candle.timeframe == "H1",
                )
            )
            .order_by(Candle.timestamp.desc())
            .limit(100)
        )
        result = await session.execute(stmt)
        rows = result.all()

        if len(rows) < 20:
            return (1.0, 1.0)

        rows = list(reversed(rows))
        highs = pd.Series([float(r[0]) for r in rows])
        lows = pd.Series([float(r[1]) for r in rows])
        closes = pd.Series([float(r[2]) for r in rows])

        atr_series = compute_atr(highs, lows, closes, length=14)
        atr_valid = atr_series.dropna()

        if atr_valid.empty:
            return (1.0, 1.0)

        current_atr = float(atr_valid.iloc[-1])
        baseline_atr = float(atr_valid.mean())

        if current_atr <= 0 or baseline_atr <= 0:
            return (1.0, 1.0)

        return (current_atr, baseline_atr)
