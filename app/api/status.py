"""Detailed /status diagnostic endpoint for operational visibility."""

import traceback
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models.candle import Candle
from app.models.signal import Signal
from app.schemas.status import SchedulerJobInfo, StatusResponse
from app.workers.scheduler import scheduler

router = APIRouter(tags=["status"])

# Track application start time for uptime calculation
_start_time: datetime = datetime.now(UTC)


@router.get("/status", response_model=StatusResponse)
async def status(
    session: AsyncSession = Depends(get_session),
) -> StatusResponse:
    """Return detailed operational diagnostics."""
    now = datetime.now(UTC)
    uptime = (now - _start_time).total_seconds()

    db_status = "connected"
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error(f"Status check -- database error: {exc}")
        db_status = "disconnected"

    scheduler_status = "running" if scheduler.running else "stopped"
    jobs: list[SchedulerJobInfo] = []
    for job in scheduler.get_jobs():
        jobs.append(
            SchedulerJobInfo(
                id=job.id,
                name=job.name,
                next_run_time=job.next_run_time,
                trigger=str(job.trigger),
            )
        )

    active_signals = 0
    try:
        result = await session.execute(
            select(func.count()).select_from(Signal).where(Signal.status == "active")
        )
        active_signals = result.scalar_one()
    except Exception as exc:
        logger.error(f"Status check -- active signals query error: {exc}")

    last_candle_fetch = None
    try:
        result = await session.execute(select(func.max(Candle.timestamp)))
        last_candle_fetch = result.scalar_one()
    except Exception as exc:
        logger.error(f"Status check -- last candle fetch query error: {exc}")

    last_signal_generated = None
    try:
        result = await session.execute(select(func.max(Signal.created_at)))
        last_signal_generated = result.scalar_one()
    except Exception as exc:
        logger.error(f"Status check -- last signal query error: {exc}")

    overall_status = (
        "ok" if db_status == "connected" and scheduler_status == "running" else "degraded"
    )

    return StatusResponse(
        status=overall_status,
        uptime_seconds=round(uptime, 1),
        database=db_status,
        scheduler=scheduler_status,
        jobs=jobs,
        active_signals=active_signals,
        last_candle_fetch=last_candle_fetch,
        last_signal_generated=last_signal_generated,
        timestamp=now,
    )


@router.post("/debug/backfill/{timeframe}")
async def debug_backfill(timeframe: str, outputsize: int = 5000):
    """Force a full historical backfill for a timeframe (ignores existing data)."""
    try:
        from app.config import get_settings
        from app.database import async_session_factory
        from app.services.candle_ingestor import CandleIngestor

        settings = get_settings()
        ingestor = CandleIngestor(api_key=settings.twelve_data_api_key)

        async with async_session_factory() as session:
            candles = await ingestor.fetch_candles("XAUUSD", timeframe, outputsize=outputsize)
            count = await ingestor.upsert_candles(session, candles)
            return {"status": "ok", "timeframe": timeframe, "fetched": len(candles), "upserted": count}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/debug/seed-strategies")
async def debug_seed_strategies():
    """Seed the strategies table with all registered strategies."""
    try:
        from app.database import async_session_factory
        from app.models.strategy import Strategy
        from sqlalchemy import select

        from app.strategies.base import BaseStrategy  # noqa: F401
        import app.strategies.liquidity_sweep  # noqa: F401
        import app.strategies.trend_continuation  # noqa: F401
        import app.strategies.breakout_expansion  # noqa: F401

        registry = BaseStrategy.get_registry()

        async with async_session_factory() as session:
            existing = await session.execute(select(Strategy))
            existing_names = {s.name for s in existing.scalars().all()}

            created = []
            for name in registry:
                if name not in existing_names:
                    session.add(Strategy(name=name, is_active=True))
                    created.append(name)

            await session.commit()

            return {"status": "ok", "created": created, "existing": list(existing_names)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/debug/create-tables")
async def debug_create_tables():
    """Create all database tables directly (bypasses Alembic)."""
    try:
        from app.database import engine
        from app.models import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        return {"status": "ok", "tables": list(Base.metadata.tables.keys())}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.get("/debug/api-test")
async def debug_api_test():
    """Test Twelve Data API connectivity."""
    import httpx
    from app.config import get_settings

    settings = get_settings()
    key = settings.twelve_data_api_key
    key_preview = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "TOO_SHORT"

    results = {"api_key_preview": key_preview}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.twelvedata.com/price",
                params={"symbol": "XAU/USD", "apikey": key},
            )
            results["price_status"] = resp.status_code
            results["price_body"] = resp.json()
    except Exception as exc:
        results["price_error"] = f"{type(exc).__name__}: {exc}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": "XAU/USD",
                    "interval": "1h",
                    "outputsize": "3",
                    "apikey": key,
                },
            )
            results["candle_status"] = resp.status_code
            body = resp.json()
            results["candle_ok"] = body.get("status") == "ok"
            results["candle_count"] = len(body.get("values", []))
    except Exception as exc:
        results["candle_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from app.database import async_session_factory
        from app.services.candle_ingestor import CandleIngestor

        ingestor = CandleIngestor(api_key=key)
        async with async_session_factory() as session:
            count = await ingestor.fetch_and_store(session, "XAUUSD", "H1")
            results["ingest_h1_count"] = count
    except Exception as exc:
        results["ingest_error"] = f"{type(exc).__name__}: {exc}"

    return results


@router.get("/debug/signal-diagnostic")
async def signal_diagnostic():
    """Run the FULL signal pipeline as a dry run and return step-by-step results."""
    import traceback as tb_mod
    from datetime import datetime, timezone

    results: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategies": [],
        "circuit_breaker": None,
        "pipeline_steps": [],
        "errors": [],
    }

    try:
        from app.database import async_session_factory
        from app.services.feedback_controller import FeedbackController
        from app.services.signal_generator import SignalGenerator
        from app.services.risk_manager import RiskManager
        from app.services.strategy_selector import StrategySelector

        async with async_session_factory() as session:
            fb = FeedbackController()
            cb_active = await fb.check_circuit_breaker(session)
            consec_losses = await fb._count_consecutive_losses(session)
            results["circuit_breaker"] = {
                "active": cb_active,
                "consecutive_losses": consec_losses,
            }
            if cb_active:
                results["pipeline_steps"].append("BLOCKED by circuit breaker")
                return results

            selector = StrategySelector()
            ranked = await selector.select_all_ranked(session)
            results["ranked_strategies"] = [
                {
                    "name": s.strategy_name,
                    "score": round(s.composite_score, 4),
                    "degraded": s.is_degraded,
                    "trades": s.total_trades,
                }
                for s in ranked
            ]
            if not ranked:
                results["pipeline_steps"].append("BLOCKED: no qualifying strategies")
                return results

            generator = SignalGenerator()
            risk_manager = RiskManager()

            for score in ranked:
                strat_info: dict = {
                    "name": score.strategy_name,
                    "pipeline_steps": [],
                }

                try:
                    candidates = await generator.generate(session, score.strategy_name)
                    strat_info["candidates_raw"] = len(candidates)
                    if not candidates:
                        strat_info["pipeline_steps"].append("generate() returned 0 candidates")
                        results["strategies"].append(strat_info)
                        continue

                    strat_info["candidate_details"] = [
                        {
                            "direction": c.direction.value,
                            "entry": str(c.entry_price),
                            "sl": str(c.stop_loss),
                            "tp1": str(c.take_profit_1),
                            "rr": str(c.risk_reward),
                            "confidence": str(c.confidence),
                        }
                        for c in candidates
                    ]

                    valid = await generator.validate(session, candidates)
                    strat_info["candidates_after_validate"] = len(valid)
                    if not valid:
                        strat_info["pipeline_steps"].append(
                            f"All {len(candidates)} candidates filtered by validate()"
                        )
                        results["strategies"].append(strat_info)
                        continue

                    valid.sort(key=lambda c: float(c.confidence), reverse=True)
                    best = valid[0]
                    strat_info["best_candidate"] = {
                        "direction": best.direction.value,
                        "entry": str(best.entry_price),
                        "confidence": str(best.confidence),
                        "rr": str(best.risk_reward),
                    }

                    from sqlalchemy import select as sa_select
                    from app.models.signal import Signal

                    active_stmt = (
                        sa_select(Signal.direction)
                        .where(Signal.status == "active")
                        .limit(1)
                    )
                    active_result = await session.execute(active_stmt)
                    active_dir = active_result.scalar_one_or_none()
                    strat_info["active_direction"] = active_dir

                    if active_dir is not None and best.direction.value != active_dir:
                        strat_info["pipeline_steps"].append(
                            f"BLOCKED: opposite direction ({best.direction.value} vs active {active_dir})"
                        )
                        results["strategies"].append(strat_info)
                        continue

                    from app.services.signal_pipeline import SignalPipeline

                    pipeline = SignalPipeline(selector, generator, risk_manager, None)
                    current_atr, baseline_atr = await pipeline._compute_atr(session)
                    strat_info["atr"] = {
                        "current": round(current_atr, 4),
                        "baseline": round(baseline_atr, 4),
                    }

                    risk_results = await risk_manager.check(
                        session,
                        [best],
                        current_atr=current_atr,
                        baseline_atr=baseline_atr,
                    )
                    for candidate, risk_result in risk_results:
                        if risk_result.approved:
                            strat_info["risk_approved"] = True
                            strat_info["position_size"] = str(risk_result.position_size)
                            strat_info["pipeline_steps"].append("WOULD GENERATE SIGNAL")
                        else:
                            strat_info["risk_approved"] = False
                            strat_info["risk_rejection"] = risk_result.rejection_reason
                            strat_info["pipeline_steps"].append(
                                f"BLOCKED by risk: {risk_result.rejection_reason}"
                            )

                    results["strategies"].append(strat_info)

                    if strat_info.get("risk_approved"):
                        results["pipeline_steps"].append(
                            f"Signal WOULD be generated from '{score.strategy_name}'"
                        )
                        break

                except Exception as e:
                    strat_info["error"] = f"{type(e).__name__}: {e}"
                    strat_info["traceback"] = tb_mod.format_exc()
                    results["strategies"].append(strat_info)
            else:
                results["pipeline_steps"].append("All strategies exhausted, no signal")

    except Exception as e:
        results["errors"].append(f"{type(e).__name__}: {e}")
        results["traceback"] = tb_mod.format_exc()

    return results


@router.api_route("/trigger/{job_name}", methods=["GET", "POST"])
async def trigger_job(job_name: str):
    """Manually trigger a job and return the result or error."""
    from app.workers.jobs import (
        job_claude_agent,
        job_detect_crypto_outcomes,
        job_detect_outcomes,
        job_fetch_candles,
        job_fetch_crypto_candles,
        job_generate_crypto_signals,
        job_generate_signals,
        job_run_backtests,
        job_send_health_digest,
    )

    job_map = {
        "refresh_candles_M15": job_fetch_candles,
        "refresh_candles_H1": job_fetch_candles,
        "refresh_candles_H4": job_fetch_candles,
        "refresh_candles_D1": job_fetch_candles,
        "run_daily_backtests": job_run_backtests,
        "run_signal_scanner": job_generate_signals,
        "check_outcomes": job_detect_outcomes,
        "fetch_crypto_candles": job_fetch_crypto_candles,
        "generate_crypto_signals": job_generate_crypto_signals,
        "check_crypto_outcomes": job_detect_crypto_outcomes,
        "send_health_digest": job_send_health_digest,
        "claude_agent": job_claude_agent,
    }

    if job_name not in job_map:
        return {"error": f"Unknown job: {job_name}", "available": list(job_map.keys())}

    try:
        await job_map[job_name]()
        return {"status": "ok", "job": job_name}
    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Manual trigger failed: {}", job_name)
        return {"status": "error", "job": job_name, "error": str(exc), "traceback": tb}


@router.get("/debug/claude-agent-status")
async def claude_agent_status():
    """Show current Claude agent configuration so you can verify env vars are set."""
    from app.config import get_settings
    s = get_settings()
    return {
        "claude_agent_enabled": s.claude_agent_enabled,
        "anthropic_api_key_set": bool(s.anthropic_api_key.strip()),
        "anthropic_api_key_preview": f"{s.anthropic_api_key[:8]}..." if s.anthropic_api_key else "NOT SET",
        "crypto_enabled": s.crypto_enabled,
        "crypto_symbols": s.crypto_symbol_list,
        "binance_testnet": s.binance_testnet,
        "binance_order_execution_enabled": s.binance_order_execution_enabled,
        "claude_agent_risk_pct": s.claude_agent_risk_pct,
        "claude_agent_daily_loss_limit": s.claude_agent_daily_loss_limit,
        "telegram_configured": bool(s.telegram_bot_token and s.telegram_chat_id),
    }
