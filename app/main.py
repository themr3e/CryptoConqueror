"""FastAPI application entry point with lifespan management."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from loguru import logger
from sqlalchemy import func, select

from app.config import get_settings
from app.database import async_sessionmaker, engine
from app.utils.logging import setup_logging
from app.workers.scheduler import scheduler
from app.workers.jobs import register_jobs
from app.api.candles import router as candles_router
from app.api.chart import router as chart_router
from app.api.dashboard import router as dashboard_router
from app.api.health import router as health_router
from app.api.settings import router as settings_router
from app.api.status import router as status_router

# ---------------------------------------------------------------------------
# Strategy → asset_class mapping
# ---------------------------------------------------------------------------
_STRATEGY_ASSET_CLASS: dict[str, str] = {
    "liquidity_sweep":       "forex",
    "trend_continuation":    "forex",
    "breakout_expansion":    "forex",
    "ema_momentum":          "forex",
    "crypto_momentum":       "crypto_futures",
    "crypto_breakout":       "crypto_futures",
}

_CRYPTO_STRATEGY_SYMBOLS = '["BTCUSDT","ETHUSDT"]'


async def bootstrap_data() -> None:
    """Seed strategies and backfill candles on first deploy.

    - XAUUSD strategies + candles: only when TWELVE_DATA_API_KEY is set.
    - Crypto strategies + candles: only when CRYPTO_ENABLED=true.
    - At least one must be enabled or bootstrap logs a warning.
    """
    settings = get_settings()

    # Import all strategies to populate the registry
    import app.strategies  # noqa: F401 — triggers all self-registrations
    from app.strategies.base import BaseStrategy
    from app.models.candle import Candle
    from app.models.strategy import Strategy

    async with async_sessionmaker() as session:
        # ── Step 1: Seed strategies with correct asset_class ──────────────
        existing_result = await session.execute(select(Strategy))
        existing_names = {s.name for s in existing_result.scalars().all()}
        registry = BaseStrategy.get_registry()

        created: list[str] = []
        for name, cls in registry.items():
            if name in existing_names:
                continue
            asset_class = getattr(cls, "ASSET_CLASS", None) or _STRATEGY_ASSET_CLASS.get(name, "forex")
            symbols = _CRYPTO_STRATEGY_SYMBOLS if asset_class == "crypto_futures" else None
            session.add(Strategy(
                name=name,
                is_active=True,
                asset_class=asset_class,
                symbols=symbols,
            ))
            created.append(name)

        if created:
            await session.commit()
            logger.info("Bootstrap: seeded {} strategies: {}", len(created), created)
        else:
            logger.info("Bootstrap: all strategies already exist")

        # ── Step 2: Backfill XAUUSD candles (only if Twelve Data key set) ──
        if settings.xauusd_enabled:
            from app.services.candle_ingestor import CandleIngestor
            ingestor = CandleIngestor(api_key=settings.twelve_data_api_key)

            for tf, min_bars in [("H1", 800), ("H4", 100), ("D1", 100)]:
                count_result = await session.execute(
                    select(func.count()).select_from(Candle).where(
                        Candle.symbol == "XAUUSD", Candle.timeframe == tf
                    )
                )
                existing_bars = count_result.scalar() or 0
                if existing_bars < min_bars:
                    logger.info("Bootstrap: backfilling XAUUSD {} ({} bars exist, need {})...",
                                tf, existing_bars, min_bars)
                    try:
                        candles = await ingestor.fetch_candles("XAUUSD", tf, outputsize=5000)
                        stored = await ingestor.upsert_candles(session, candles)
                        logger.info("Bootstrap: backfilled {} XAUUSD {} candles", stored, tf)
                    except Exception:
                        logger.opt(exception=True).warning("Bootstrap: XAUUSD {} backfill failed", tf)
                else:
                    logger.info("Bootstrap: XAUUSD {} OK ({} bars)", tf, existing_bars)
        else:
            logger.info("Bootstrap: XAUUSD disabled (no TWELVE_DATA_API_KEY) — skipping Gold candles")

        # ── Step 3: Backfill crypto candles (only if CRYPTO_ENABLED=true) ──
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
            logger.info("Bootstrap: crypto disabled (CRYPTO_ENABLED=false) — skipping crypto candles")

        # ── Warning if nothing is enabled ─────────────────────────────────
        if not settings.xauusd_enabled and not settings.crypto_enabled:
            logger.warning(
                "Bootstrap: BOTH XAUUSD and crypto are disabled! "
                "Set TWELVE_DATA_API_KEY for Gold or CRYPTO_ENABLED=true for crypto."
            )

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
