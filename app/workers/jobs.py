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
from app.services.self_improver import SelfImprover
from app.services.strategy_researcher import StrategyResearcher
from app.services.telegram_commander import TelegramCommander
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
    """Run rolling backtests for all active strategies and persist results to DB.

    The StrategySelector queries BacktestResult rows to decide which strategy
    to use for signal generation.  Without persisted rows, no signals are ever
    generated.  This job runs daily and writes fresh BacktestResult records so
    the selector always has up-to-date data.
    """
    settings = get_settings()
    if not settings.crypto_enabled:
        return

    logger.info("[Job] run_backtests started")
    runner = _get_backtest_runner()
    async with async_sessionmaker() as session:
        try:
            import pandas as pd
            from datetime import datetime, timezone, timedelta
            from sqlalchemy import and_, select
            from app.models.backtest_result import BacktestResult
            from app.models.candle import Candle
            from app.models.strategy import Strategy

            # Run on up to 5 symbols and aggregate metrics so strategies
            # accumulate enough trades to pass MIN_TRADES_QUALIFY.
            sample_symbols = settings.crypto_symbol_list[:5] if settings.crypto_symbol_list else ["BTCUSDT"]

            # Collect (metrics, trades) per strategy+window across all symbols
            from collections import defaultdict
            from app.services.metrics_calculator import MetricsCalculator
            from app.strategies.base import BaseStrategy
            aggregated: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

            for symbol in sample_symbols:
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
                    continue

                df = pd.DataFrame([{
                    "timestamp": c.timestamp,
                    "open":   float(c.open),
                    "high":   float(c.high),
                    "low":    float(c.low),
                    "close":  float(c.close),
                    "volume": float(c.volume) if c.volume is not None else 0.0,
                } for c in candles_orm]).set_index("timestamp")
                df.attrs["symbol"] = symbol

                sym_results = runner.run_all_strategies(df)
                for strategy_name, window_map in sym_results.items():
                    for window_days, (_metrics, trades) in window_map.items():
                        aggregated[strategy_name][window_days].extend(trades)

            # Recompute metrics from all-symbol aggregated trades
            metrics_calc = MetricsCalculator()
            backtest_results: dict[str, dict[int, tuple]] = {}
            for strategy_name, window_map in aggregated.items():
                backtest_results[strategy_name] = {}
                for window_days, all_trades in window_map.items():
                    metrics = metrics_calc.compute(all_trades)
                    backtest_results[strategy_name][window_days] = (metrics, all_trades)
                    logger.info(
                        "[Job] run_backtests: {} window={}d trades={} wr={} pf={}",
                        strategy_name, window_days, metrics.total_trades,
                        metrics.win_rate, metrics.profit_factor,
                    )

            # ── Persist BacktestResult records ────────────────────────────────
            now = datetime.now(timezone.utc)
            saved = 0
            for strategy_name, window_results in backtest_results.items():
                strat_row = await session.execute(
                    select(Strategy).where(Strategy.name == strategy_name)
                )
                strategy_orm = strat_row.scalar_one_or_none()
                if strategy_orm is None:
                    logger.warning(
                        "[Job] run_backtests: strategy '{}' not found in DB — skipping",
                        strategy_name,
                    )
                    continue

                for window_days, (metrics, _trades) in window_results.items():
                    session.add(BacktestResult(
                        strategy_id=strategy_orm.id,
                        timeframe="H1",
                        window_days=window_days,
                        start_date=now - timedelta(days=window_days),
                        end_date=now,
                        win_rate=metrics.win_rate,
                        profit_factor=metrics.profit_factor,
                        sharpe_ratio=metrics.sharpe_ratio,
                        max_drawdown=metrics.max_drawdown,
                        expectancy=metrics.expectancy,
                        total_trades=metrics.total_trades,
                        is_walk_forward=False,
                    ))
                    saved += 1

            await session.commit()

            total_trades = sum(
                len(trades)
                for strat_results in backtest_results.values()
                for _metrics, trades in strat_results.values()
            )
            logger.info(
                "[Job] run_backtests: {} strategy(ies), {} simulated trade(s), "
                "{} BacktestResult row(s) saved",
                len(backtest_results), total_trades, saved,
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


async def job_send_daily_report() -> None:
    """Send a full daily trading report via Telegram at 08:00 UTC."""
    logger.info("[Job] daily_report started")
    _s = get_settings()
    notifier = TelegramNotifier(bot_token=_s.telegram_bot_token or "", chat_id=_s.telegram_chat_id or "")

    async with async_sessionmaker() as session:
        try:
            from sqlalchemy import select, func
            from app.models.signal import Signal
            from app.models.outcome import Outcome
            from app.models.claude_decision import ClaudeDecision
            from datetime import timedelta

            now   = datetime.now(timezone.utc)
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # ── Today's outcomes ────────────────────────────────────────────
            outcomes_result = await session.execute(
                select(Outcome, Signal.symbol, Signal.direction)
                .join(Signal, Outcome.signal_id == Signal.id)
                .where(Outcome.created_at >= today)
                .order_by(Outcome.pnl_usdt.desc())
            )
            rows = outcomes_result.all()

            total_trades  = len(rows)
            wins          = sum(1 for r in rows if r[0].result in ("tp1_hit", "tp2_hit"))
            losses        = total_trades - wins
            total_pnl     = sum(float(r[0].pnl_usdt or 0) for r in rows)
            win_rate      = (wins / total_trades * 100) if total_trades else 0

            best_trade  = rows[0]  if rows else None
            worst_trade = rows[-1] if rows else None

            # ── Open positions ───────────────────────────────────────────────
            open_result = await session.execute(
                select(Signal).where(Signal.status == "active")
            )
            open_signals = open_result.scalars().all()

            # ── 7-day win rate ───────────────────────────────────────────────
            week_ago = now - timedelta(days=7)
            week_result = await session.execute(
                select(Outcome).where(Outcome.created_at >= week_ago)
            )
            week_outcomes = week_result.scalars().all()
            wins_7d  = sum(1 for o in week_outcomes if o.result in ("tp1_hit", "tp2_hit"))
            total_7d = len(week_outcomes)
            wr_7d    = f"{wins_7d}/{total_7d} ({wins_7d/total_7d*100:.0f}%)" if total_7d else "no data"
            pnl_7d   = sum(float(o.pnl_usdt or 0) for o in week_outcomes)

            # ── Latest self-improvement insight ──────────────────────────────
            improver = SelfImprover()
            insight  = await improver.get_latest_insight(session)

            # ── Build message ────────────────────────────────────────────────
            pnl_emoji = "📈" if total_pnl >= 0 else "📉"
            lines = [
                f"{pnl_emoji} <b>Daily Report — {now.strftime('%Y-%m-%d')}</b>",
                "",
                f"<b>Trades today:</b> {total_trades} ({wins}W / {losses}L)",
                f"<b>Win rate today:</b> {win_rate:.0f}%",
                f"<b>Today's P&amp;L:</b> ${total_pnl:+.2f}",
                "",
                f"<b>7-day win rate:</b> {wr_7d}",
                f"<b>7-day P&amp;L:</b> ${pnl_7d:+.2f}",
                "",
            ]

            if best_trade and float(best_trade[0].pnl_usdt or 0) > 0:
                lines.append(
                    f"🏆 <b>Best trade:</b> {best_trade[1]} {best_trade[2]} "
                    f"→ ${float(best_trade[0].pnl_usdt):+.2f}"
                )
            if worst_trade and float(worst_trade[0].pnl_usdt or 0) < 0:
                lines.append(
                    f"💀 <b>Worst trade:</b> {worst_trade[1]} {worst_trade[2]} "
                    f"→ ${float(worst_trade[0].pnl_usdt):+.2f}"
                )

            if open_signals:
                lines.append("")
                lines.append(f"<b>Open positions:</b> {len(open_signals)}")
                for s in open_signals[:3]:
                    lines.append(
                        f"  {'📈' if s.direction == 'BUY' else '📉'} {s.symbol} "
                        f"@ {float(s.entry_price):.4f}"
                    )

            if insight:
                lines.append("")
                # Extract just the summary line from insight block
                for line in insight.split("\n"):
                    if "Summary:" in line or "✏️" in line or "⛔" in line or "🎯" in line:
                        lines.append(line.strip())

            lines.append("")
            lines.append("Send /status for live snapshot anytime.")

            await notifier._send_message("\n".join(lines))
            logger.info("[Job] daily_report sent — {} trades, ${:+.2f} P&L", total_trades, total_pnl)

        except Exception:
            logger.opt(exception=True).error("[Job] daily_report failed")


async def job_strategy_research() -> None:
    """Weekly autonomous strategy R&D cycle.

    Claude reviews recent trade history, proposes a new strategy,
    runs a blind walk-forward backtest, and either auto-integrates
    (win rate >= 75%) or notifies the operator for approval.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        return

    logger.info("[Job] strategy_research started")
    _s = get_settings()
    notifier = TelegramNotifier(bot_token=_s.telegram_bot_token or "", chat_id=_s.telegram_chat_id or "")

    async with async_sessionmaker() as session:
        try:
            researcher = StrategyResearcher()
            summary = await researcher.run(session)
            await notifier._send_message(f"🔬 <b>Weekly Strategy Research</b>\n\n{summary}")
            logger.info("[Job] strategy_research complete")
        except Exception:
            logger.opt(exception=True).error("[Job] strategy_research failed")


async def job_self_improve() -> None:
    """Trigger self-improvement analysis after every 10 closed trades."""
    async with async_sessionmaker() as session:
        try:
            improver = SelfImprover()
            ran = await improver.maybe_analyze(session)
            if ran:
                logger.info("[Job] self_improve: analysis completed")
        except Exception:
            logger.opt(exception=True).error("[Job] self_improve failed")


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
            for tf in ["M15", "M30", "H1", "H4", "D1"]:
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

                # Close actual Binance positions and cancel dangling SL/TP orders
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
                                # Cancel any dangling SL/TP orders first
                                cancelled = await executor.cancel_signal_orders(
                                    session, signal.id, signal.symbol
                                )
                                if cancelled:
                                    logger.info(
                                        "[Job] Cancelled {} dangling order(s) for signal {}",
                                        cancelled, signal.id,
                                    )
                                # Close the actual Binance position so DB outcome
                                # matches reality (prevents stale open positions
                                # from corrupting win-rate data)
                                pos_size = await executor.get_open_position_size(signal.symbol)
                                if pos_size != 0.0:
                                    closed = await executor.close_position(signal.symbol, pos_size)
                                    if closed:
                                        logger.info(
                                            "[Job] Closed Binance position for signal {} {} (size={})",
                                            signal.id, signal.symbol, pos_size,
                                        )
                                    else:
                                        logger.warning(
                                            "[Job] Failed to close Binance position for signal {} {}",
                                            signal.id, signal.symbol,
                                        )
                        except Exception:
                            logger.opt(exception=True).warning(
                                "[Job] Failed to close/cancel orders for outcome {}", outcome.id
                            )
        except Exception:
            logger.opt(exception=True).error("[Job] detect_crypto_outcomes failed")


async def job_reconcile_unexecuted_signals() -> None:
    """Every 15 min: find active signals with no TradeOrder and execute them on Binance.

    Safety net for signals that arrived via the TradingView webhook before the
    webhook was updated to call execute_signal() directly, or in case the
    executor crashed mid-flight and left the signal orphaned.
    """
    settings = get_settings()
    if not settings.crypto_enabled or not settings.binance_order_execution_enabled:
        return

    from sqlalchemy import select, not_, exists
    from app.models.signal import Signal
    from app.models.trade_order import TradeOrder
    from datetime import timedelta

    async with async_sessionmaker() as session:
        try:
            # Active signals that have zero TradeOrder records
            orphaned_stmt = (
                select(Signal)
                .where(
                    Signal.status == "active",
                    not_(
                        exists(
                            select(TradeOrder.id).where(TradeOrder.signal_id == Signal.id)
                        )
                    ),
                    # Only retry if signal was created within last 2 hours (not ancient)
                    Signal.created_at >= datetime.now(timezone.utc) - timedelta(hours=2),
                )
            )
            result = await session.execute(orphaned_stmt)
            orphaned = result.scalars().all()

            if not orphaned:
                return

            logger.warning(
                "[Reconcile] Found {} active signal(s) with no Binance order — executing now",
                len(orphaned),
            )
            executor = _get_binance_executor()
            _s = get_settings()
            notifier = TelegramNotifier(bot_token=_s.telegram_bot_token or "", chat_id=_s.telegram_chat_id or "")

            for signal in orphaned:
                exec_result = await executor.execute_signal(session, signal)
                if exec_result.success:
                    logger.info(
                        "[Reconcile] Executed orphaned signal #{} {} {}",
                        signal.id, signal.direction, signal.symbol,
                    )
                    await notifier._send_message(
                        f"⚠️ <b>Reconcile:</b> Late Binance order placed\n"
                        f"Signal #{signal.id} {signal.symbol} {signal.direction} — was unexecuted"
                    )
                else:
                    logger.error(
                        "[Reconcile] Failed to execute orphaned signal #{} — {}",
                        signal.id, exec_result.error_message,
                    )
                    await notifier._send_message(
                        f"🚨 <b>Reconcile FAILED:</b> Signal #{signal.id} {signal.symbol} "
                        f"{signal.direction} has no Binance order and execution failed: "
                        f"{exec_result.error_message}"
                    )
        except Exception:
            logger.opt(exception=True).error("[Job] reconcile_unexecuted_signals failed")


async def job_iceberg_watchdog() -> None:
    """Restart any iceberg scanner threads that crashed."""
    from app.services.iceberg_monitor import iceberg_monitor
    iceberg_monitor.check_threads()


async def job_telegram_commander() -> None:
    """Poll Telegram for incoming commands every 15 seconds."""
    commander = TelegramCommander()
    async with async_sessionmaker() as session:
        try:
            await commander.poll_and_handle(session)
        except Exception:
            logger.opt(exception=True).error("[Job] telegram_commander failed")


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
        - detect_crypto_outcomes  : every 2 minutes

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
        job_send_daily_report,
        trigger="cron",
        hour=8,
        minute=0,
        id="daily_report",
        name="Send full daily trading report",
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
        minutes=2,
        id="detect_crypto_outcomes",
        name="Detect crypto signal outcomes",
    )

    # Weekly strategy research (every Sunday at 03:00 UTC)
    scheduler.add_job(
        job_strategy_research,
        trigger="cron",
        day_of_week="sun",
        hour=3,
        minute=0,
        id="strategy_research",
        name="Weekly autonomous strategy R&D",
    )

    # Self-improvement loop (runs every 2 min, triggers analysis every 10th trade)
    scheduler.add_job(
        job_self_improve,
        trigger="interval",
        minutes=2,
        id="self_improve",
        name="Self-improvement analysis",
    )

    # Signal reconciliation — catch signals with no Binance order (every 15 min)
    scheduler.add_job(
        job_reconcile_unexecuted_signals,
        trigger="interval",
        minutes=15,
        id="reconcile_unexecuted_signals",
        name="Reconcile active signals missing Binance orders",
    )

    # Iceberg monitor watchdog — restarts crashed scanner threads (every 5 min)
    scheduler.add_job(
        job_iceberg_watchdog,
        trigger="interval",
        minutes=5,
        id="iceberg_watchdog",
        name="Iceberg scanner thread watchdog",
    )

    # Telegram two-way command interface (every 15 seconds)
    scheduler.add_job(
        job_telegram_commander,
        trigger="interval",
        seconds=15,
        id="telegram_commander",
        name="Telegram command handler",
    )

    # Claude autonomous agent (every 15 minutes — aligned with M15 candle close)
    scheduler.add_job(
        job_claude_agent,
        trigger="interval",
        minutes=15,
        id="claude_agent",
        name="Claude autonomous trading agent",
    )

    logger.info("Registered {} background jobs", len(scheduler.get_jobs()))
