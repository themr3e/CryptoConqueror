"""Background job definitions.

All recurring tasks (candle ingestion, signal generation, outcome detection,
backtesting, performance tracking, data retention) are defined here and
registered with the APScheduler instance.

Exports:
    register_jobs -- registers all jobs on the given scheduler
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from app.config import get_settings
from app.database import async_session_factory as async_sessionmaker
from app.services.backtester import BacktestRunner
from app.services.binance_executor import BinanceExecutor
from app.services.crypto_candle_ingestor import CryptoCandleIngestor
from app.services.crypto_fee_model import CryptoFeeModel
from app.services.crypto_outcome_detector import CryptoOutcomeDetector
from app.services.data_retention import DataRetentionService
from app.services.failure_tracker import FailureTracker
from app.services.feedback_controller import FeedbackController
from app.services.param_optimizer import ParamOptimizer
from app.services.performance_tracker import PerformanceTracker
from app.services.risk_manager import RiskManager
from app.services.claude_trade_agent import ClaudeTradeAgent
from app.services.signal_generator import SignalGenerator
from app.services.signal_pipeline import SignalPipeline
from app.services.strategy_selector import StrategySelector
from app.services.telegram_notifier import TelegramNotifier
from app.services.walk_forward import WalkForwardValidator


# Lazy-initialised singletons (re-used across job invocations)
_signal_pipeline: SignalPipeline | None = None
_crypto_candle_ingestor: CryptoCandleIngestor | None = None
_crypto_outcome_detector: CryptoOutcomeDetector | None = None
_binance_executor: BinanceExecutor | None = None
_backtest_runner: BacktestRunner | None = None
_param_optimizer: ParamOptimizer | None = None
_performance_tracker: PerformanceTracker | None = None
_data_retention: DataRetentionService | None = None
_feedback_controller: FeedbackController | None = None
_failure_tracker: FailureTracker = FailureTracker()


def _get_signal_pipeline() -> SignalPipeline:
    global _signal_pipeline
    if _signal_pipeline is None:
        selector = StrategySelector()
        generator = SignalGenerator()
        risk_manager = RiskManager()
        _signal_pipeline = SignalPipeline(
            selector=selector,
            generator=generator,
            risk_manager=risk_manager,
        )
    return _signal_pipeline


def _get_backtest_runner() -> BacktestRunner:
    global _backtest_runner
    if _backtest_runner is None:
        _backtest_runner = BacktestRunner()
    return _backtest_runner


def _get_param_optimizer() -> ParamOptimizer:
    global _param_optimizer
    if _param_optimizer is None:
        runner = _get_backtest_runner()
        wf_validator = WalkForwardValidator(runner=runner)
        _param_optimizer = ParamOptimizer(runner=runner, wf_validator=wf_validator)
    return _param_optimizer


def _get_performance_tracker() -> PerformanceTracker:
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker


def _get_data_retention() -> DataRetentionService:
    global _data_retention
    if _data_retention is None:
        _data_retention = DataRetentionService()
    return _data_retention


def _get_binance_executor() -> BinanceExecutor:
    global _binance_executor
    if _binance_executor is None:
        _binance_executor = BinanceExecutor()
    return _binance_executor


def _get_crypto_candle_ingestor() -> CryptoCandleIngestor:
    global _crypto_candle_ingestor
    if _crypto_candle_ingestor is None:
        _crypto_candle_ingestor = CryptoCandleIngestor()
    return _crypto_candle_ingestor


def _get_crypto_outcome_detector() -> CryptoOutcomeDetector:
    global _crypto_outcome_detector
    if _crypto_outcome_detector is None:
        _s = get_settings()
        notifier = TelegramNotifier(bot_token=_s.telegram_bot_token or "", chat_id=_s.telegram_chat_id or "")
        fee_model = CryptoFeeModel()
        perf_tracker = PerformanceTracker()
        _crypto_outcome_detector = CryptoOutcomeDetector(
            crypto_ingestor=_get_crypto_candle_ingestor(),
            fee_model=fee_model,
            notifier=notifier,
            perf_tracker=perf_tracker,
        )
    return _crypto_outcome_detector


def _get_feedback_controller() -> FeedbackController:
    global _feedback_controller
    if _feedback_controller is None:
        _feedback_controller = FeedbackController()
    return _feedback_controller


# ---------------------------------------------------------------------------
# Job implementations
# ---------------------------------------------------------------------------

async def job_run_backtests() -> None:
    """Run rolling backtests for all active strategies."""
    settings = get_settings()
    if not settings.crypto_enabled:
        return

    logger.info("[Job] run_backtests started")
    runner = _get_backtest_runner()
    async with async_sessionmaker() as session:
        try:
            import pandas as pd
            from sqlalchemy import and_, select
            from app.models.candle import Candle

            symbol = settings.crypto_symbol_list[0] if settings.crypto_symbol_list else "BTCUSDT"
            stmt = (
                select(Candle)
                .where(and_(Candle.symbol == symbol, Candle.timeframe == "H1"))
                .order_by(Candle.timestamp.asc())
                .limit(2000)
            )
            result = await session.execute(stmt)
            candles_orm = result.scalars().all()

            if not candles_orm:
                logger.warning("[Job] run_backtests: no H1 candles for {} — skipping", symbol)
                return

            df = pd.DataFrame([{
                "timestamp": c.timestamp,
                "open":   float(c.open),
                "high":   float(c.high),
                "low":    float(c.low),
                "close":  float(c.close),
                "volume": float(c.volume) if c.volume is not None else 0.0,
            } for c in candles_orm]).set_index("timestamp")
            df.attrs["symbol"] = symbol

            results = runner.run_all_strategies(df)
            total_trades = sum(
                len(trades)
                for strat_results in results.values()
                for _metrics, trades in strat_results.values()
            )
            logger.info(
                "[Job] run_backtests: {} strategy(ies), {} simulated trade(s)",
                len(results), total_trades,
            )
        except Exception:
            logger.opt(exception=True).error("[Job] run_backtests failed")


async def job_update_performance() -> None:
    """Update strategy performance records (7d and 30d)."""
    logger.info("[Job] update_performance started")
    tracker = _get_performance_tracker()
    async with async_sessionmaker() as session:
        try:
            await tracker.update_all(session)
            logger.info("[Job] update_performance complete")
        except Exception:
            logger.opt(exception=True).error("[Job] update_performance failed")


async def job_data_retention() -> None:
    """Prune old candle and backtest data per retention policy."""
    logger.info("[Job] data_retention started")
    retention = _get_data_retention()
    async with async_sessionmaker() as session:
        try:
            summary = await retention.run(session)
            logger.info("[Job] data_retention: {}", summary)
        except Exception:
            logger.opt(exception=True).error("[Job] data_retention failed")


async def job_send_health_digest() -> None:
    """Send daily health digest via Telegram."""
    logger.info("[Job] health_digest started")
    _s = get_settings()
    notifier = TelegramNotifier(bot_token=_s.telegram_bot_token or "", chat_id=_s.telegram_chat_id or "")
    async with async_sessionmaker() as session:
        try:
            from sqlalchemy import select, func, and_
            from app.models.signal import Signal
            from app.models.outcome import Outcome
            from datetime import timedelta

            now = datetime.now(timezone.utc)
            since = now - timedelta(hours=24)

            sig_count = await session.scalar(
                select(func.count()).select_from(Signal)
                .where(Signal.created_at >= since)
            ) or 0

            outcome_count = await session.scalar(
                select(func.count()).select_from(Outcome)
                .where(Outcome.created_at >= since)
            ) or 0

            await notifier.notify_health_digest(stats={
                "active_signals": sig_count,
                "outcomes_today": outcome_count,
            })
        except Exception:
            logger.opt(exception=True).error("[Job] health_digest failed")


# ---------------------------------------------------------------------------
# Crypto jobs (guarded by CRYPTO_ENABLED setting)
# ---------------------------------------------------------------------------

async def job_fetch_crypto_candles() -> None:
    """Fetch and store latest candles for all configured crypto symbols."""
    settings = get_settings()
    if not settings.crypto_enabled:
        return

    logger.info("[Job] fetch_crypto_candles started")
    ingestor = _get_crypto_candle_ingestor()
    symbols = settings.crypto_symbol_list

    async with async_sessionmaker() as session:
        for symbol in symbols:
            for tf in ["M15", "H1", "H4", "D1"]:
                try:
                    stored = await ingestor.fetch_and_store(session, symbol, tf)
                    logger.info("[Job] fetch_crypto_candles: {} {} {} candles stored", stored, symbol, tf)
                    _failure_tracker.record_success("crypto_candle_fetch")
                except Exception:
                    logger.opt(exception=True).error(
                        "[Job] fetch_crypto_candles failed for {} {}", symbol, tf
                    )
                    _failure_tracker.record_failure("crypto_candle_fetch")


async def job_generate_crypto_signals() -> None:
    """Run the signal pipeline for all configured crypto symbols.

    After generating signals, automatically executes orders on Binance
    Futures (testnet or mainnet) if API keys are configured.
    """
    settings = get_settings()
    if not settings.crypto_enabled:
        return

    logger.info("[Job] generate_crypto_signals started")
    pipeline = _get_signal_pipeline()
    symbols = settings.crypto_symbol_list

    async with async_sessionmaker() as session:
        for symbol in symbols:
            try:
                signals = await pipeline.run(session, symbol=symbol)
                logger.info("[Job] generate_crypto_signals: {} signal(s) for {}", len(signals), symbol)

                # Auto-execute orders if API keys are configured
                if signals and settings.binance_order_execution_enabled:
                    executor = _get_binance_executor()
                    env = "TESTNET" if settings.binance_testnet else "MAINNET"
                    for signal in signals:
                        try:
                            result = await executor.execute_signal(session, signal)
                            if result.success:
                                logger.info(
                                    "[Job] Order executed on {} — signal {} {} {}",
                                    env, signal.id, signal.direction, signal.symbol,
                                )
                            else:
                                logger.warning(
                                    "[Job] Order execution failed — signal {}: {}",
                                    signal.id, result.error_message,
                                )
                        except Exception:
                            logger.opt(exception=True).error(
                                "[Job] Order execution error for signal {}", signal.id
                            )
                elif signals and not settings.binance_order_execution_enabled:
                    logger.info(
                        "[Job] {} signal(s) generated but no Binance API keys configured "
                        "— manual trading mode (check dashboard/Telegram for alerts)",
                        len(signals),
                    )

            except Exception:
                logger.opt(exception=True).error("[Job] generate_crypto_signals failed for {}", symbol)


async def job_detect_crypto_outcomes() -> None:
    """Check active crypto signals against Binance mark price and record outcomes.

    When an outcome is detected and API keys are configured, cancels any
    remaining open SL/TP orders on Binance to avoid accidental re-fills.
    """
    settings = get_settings()
    if not settings.crypto_enabled:
        return

    logger.info("[Job] detect_crypto_outcomes started")
    detector = _get_crypto_outcome_detector()
    symbols = settings.crypto_symbol_list

    async with async_sessionmaker() as session:
        try:
            outcomes = await detector.check_active_signals(session, crypto_symbols=symbols)
            if outcomes:
                logger.info("[Job] detect_crypto_outcomes: {} outcome(s) recorded", len(outcomes))

                # Cancel any dangling SL/TP orders for closed signals
                if settings.binance_order_execution_enabled:
                    executor = _get_binance_executor()
                    from app.models.signal import Signal
                    from sqlalchemy import select
                    for outcome in outcomes:
                        try:
                            signal_result = await session.execute(
                                select(Signal).where(Signal.id == outcome.signal_id)
                            )
                            signal = signal_result.scalar_one_or_none()
                            if signal:
                                cancelled = await executor.cancel_signal_orders(
                                    session, signal.id, signal.symbol
                                )
                                if cancelled:
                                    logger.info(
                                        "[Job] Cancelled {} dangling order(s) for signal {}",
                                        cancelled, signal.id,
                                    )
                        except Exception:
                            logger.opt(exception=True).warning(
                                "[Job] Failed to cancel orders for outcome {}", outcome.id
                            )
        except Exception:
            logger.opt(exception=True).error("[Job] detect_crypto_outcomes failed")


async def job_claude_agent() -> list:
    """Run Claude autonomous trading agent for all crypto symbols."""
    settings = get_settings()
    if not settings.claude_agent_enabled:
        return [{"info": "CLAUDE_AGENT_ENABLED is false — skipping"}]
    if not settings.anthropic_api_key:
        logger.warning("[ClaudeAgent] ANTHROPIC_API_KEY not set — skipping")
        return [{"info": "ANTHROPIC_API_KEY not set — skipping"}]

    logger.info("[Job] claude_agent started (batch mode — 1 API call for {} symbols)", len(settings.crypto_symbol_list))
    agent = ClaudeTradeAgent()
    decisions = []
    async with async_sessionmaker() as session:
        try:
            batch = await agent.run_batch(session, settings.crypto_symbol_list)
            for decision in batch:
                logger.info(
                    "[Job] claude_agent: {} → {} (confidence: {}%)",
                    decision.symbol, decision.action, decision.confidence,
                )
                decisions.append({
                    "symbol": decision.symbol,
                    "action": decision.action,
                    "confidence": float(decision.confidence or 0),
                    "reasoning": decision.reasoning,
                    "executed": decision.executed,
                    "error": decision.execution_error,
                })
        except Exception as exc:
            logger.opt(exception=True).error("[Job] claude_agent batch failed")
            decisions.append({"error": str(exc)})
    return decisions


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_jobs(scheduler: AsyncIOScheduler) -> None:
    """Register all background jobs on the scheduler.

    Job schedule:
        - run_backtests           : daily at 02:00 UTC
        - update_performance      : daily at 01:00 UTC
        - data_retention          : daily at 04:00 UTC
        - health_digest           : daily at 08:00 UTC

        Crypto (active only when CRYPTO_ENABLED=true):
        - fetch_crypto_candles    : every 15 minutes (Binance Futures)
        - generate_crypto_signals : every hour at :05
        - detect_crypto_outcomes  : every 5 minutes

        Claude agent (active only when CLAUDE_AGENT_ENABLED=true):
        - claude_agent            : every 30 minutes
    """
    scheduler.add_job(
        job_run_backtests,
        trigger="cron",
        hour=2,
        minute=0,
        id="run_backtests",
        name="Run strategy backtests",
    )
    scheduler.add_job(
        job_update_performance,
        trigger="cron",
        hour=1,
        minute=0,
        id="update_performance",
        name="Update strategy performance",
    )
    scheduler.add_job(
        job_data_retention,
        trigger="cron",
        hour=4,
        minute=0,
        id="data_retention",
        name="Data retention cleanup",
    )
    scheduler.add_job(
        job_send_health_digest,
        trigger="cron",
        hour=8,
        minute=0,
        id="health_digest",
        name="Send health digest",
    )

    # ── Crypto jobs (no-op when CRYPTO_ENABLED=false) ─────────────────────
    scheduler.add_job(
        job_fetch_crypto_candles,
        trigger="interval",
        minutes=15,
        id="fetch_crypto_candles",
        name="Fetch crypto candles (Binance Futures)",
    )
    scheduler.add_job(
        job_generate_crypto_signals,
        trigger="cron",
        minute=5,
        id="generate_crypto_signals",
        name="Generate crypto trading signals",
    )
    scheduler.add_job(
        job_detect_crypto_outcomes,
        trigger="interval",
        minutes=5,
        id="detect_crypto_outcomes",
        name="Detect crypto signal outcomes",
    )

    # Claude autonomous agent (every 30 minutes when enabled)
    scheduler.add_job(
        job_claude_agent,
        trigger="interval",
        minutes=30,
        id="claude_agent",
        name="Claude autonomous trading agent",
    )

    logger.info("Registered {} background jobs", len(scheduler.get_jobs()))
