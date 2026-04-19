"""FastAPI application entry point with lifespan management."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import json

from fastapi import FastAPI
from loguru import logger
from sqlalchemy import func, select

from app.config import get_settings
from app.database import async_session_factory as async_sessionmaker, engine
from app.utils.logging import setup_logging
from app.workers.scheduler import scheduler
from app.workers.jobs import register_jobs
from app.api.candles import router as candles_router
from app.api.chart import router as chart_router
from app.api.dashboard import router as dashboard_router
from app.api.health import router as health_router
from app.api.settings import router as settings_router
from app.api.status import router as status_router
from app.api.webhook import router as webhook_router

async def bootstrap_data() -> None:
    """Seed strategies and backfill crypto candles on first deploy."""
    settings = get_settings()

    # Import all strategies to populate the registry
    import app.strategies  # noqa: F401 — triggers all self-registrations
    from app.strategies.base import BaseStrategy
    from app.models.candle import Candle
    from app.models.strategy import Strategy

    async with async_sessionmaker() as session:
        # ── Step 1: Seed strategies ────────────────────────────────────────
        existing_result = await session.execute(select(Strategy))
        existing_names = {s.name for s in existing_result.scalars().all()}
        registry = BaseStrategy.get_registry()

        created: list[str] = []
        for name, cls in registry.items():
            if name in existing_names:
                continue
            asset_class = getattr(cls, "ASSET_CLASS", "crypto_futures")
            symbols_json = json.dumps(settings.crypto_symbol_list)
            session.add(Strategy(
                name=name,
                is_active=True,
                asset_class=asset_class,
                symbols=symbols_json,
            ))
            created.append(name)

        if created:
            await session.commit()
            logger.info("Bootstrap: seeded {} strategies: {}", len(created), created)
        else:
            logger.info("Bootstrap: all strategies already exist")

        # ── Step 2: Backfill crypto candles ───────────────────────────────
        if settings.crypto_enabled:
            from app.services.crypto_candle_ingestor import CryptoCandleIngestor
            crypto_ingestor = CryptoCandleIngestor()

            for symbol in settings.crypto_symbol_list:
                for tf, min_bars in [("H1", 800), ("H4", 100), ("D1", 100)]:
                    count_result = await session.execute(
                        select(func.count()).select_from(Candle).where(
                            Candle.symbol == symbol, Candle.timeframe == tf
                        )
                    )
                    existing_bars = count_result.scalar() or 0
                    if existing_bars < min_bars:
                        logger.info("Bootstrap: backfilling {} {} ({} bars exist)...",
                                    symbol, tf, existing_bars)
                        try:
                            stored = await crypto_ingestor.fetch_and_store(session, symbol, tf, limit=1500)
                            logger.info("Bootstrap: backfilled {} {} {} candles", stored, symbol, tf)
                        except Exception:
                            logger.opt(exception=True).warning(
                                "Bootstrap: {} {} backfill failed", symbol, tf
                            )
                    else:
                        logger.info("Bootstrap: {} {} OK ({} bars)", symbol, tf, existing_bars)
        else:
            logger.warning("Bootstrap: crypto disabled (CRYPTO_ENABLED=false) — set CRYPTO_ENABLED=true")

    # ── Step 3: Run backtests if no results exist yet ─────────────────────
    # job_run_backtests only fires at 02:00 UTC; on a fresh deploy the
    # BacktestResult table is empty, so StrategySelector never selects a
    # strategy and zero signals are generated.  Run it once at startup.
    if settings.crypto_enabled:
        from app.models.backtest_result import BacktestResult
        from app.workers.jobs import job_run_backtests
        async with async_sessionmaker() as check_session:
            count = await check_session.scalar(
                select(func.count()).select_from(BacktestResult)
            ) or 0
        logger.info("Bootstrap: running backtest ({} existing rows)...", count)
        try:
            await job_run_backtests()
            logger.info("Bootstrap: backtest complete")
        except Exception:
            logger.opt(exception=True).warning("Bootstrap: backtest failed")

    logger.info("Bootstrap: data initialization complete")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: setup on startup, teardown on shutdown."""
    import asyncio

    # Configure structured logging first so all startup logs are formatted
    setup_logging()

    logger.info("Starting QuantLive application...")

    # Start scheduler immediately so the server can accept healthcheck requests
    register_jobs(scheduler)
    scheduler.start()

    # Start iceberg monitor background threads (one per coin, daemon threads)
    from app.services.iceberg_monitor import iceberg_monitor
    iceberg_monitor.start(get_settings().crypto_symbol_list)

    # Run bootstrap in the background — does NOT block server startup
    async def _bootstrap_safe() -> None:
        try:
            await bootstrap_data()
        except Exception:
            logger.opt(exception=True).error("Bootstrap failed — continuing anyway")

    asyncio.create_task(_bootstrap_safe())

    logger.info("QuantLive application started — scheduler running")

    yield

    # Graceful shutdown
    from app.services.iceberg_monitor import iceberg_monitor
    iceberg_monitor.stop()
    scheduler.shutdown(wait=False)
    await engine.dispose()
    logger.info("QuantLive application stopped")


app = FastAPI(
    title="QuantLive",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(status_router)
app.include_router(candles_router)
app.include_router(chart_router)
app.include_router(dashboard_router)
app.include_router(settings_router)
app.include_router(webhook_router)
