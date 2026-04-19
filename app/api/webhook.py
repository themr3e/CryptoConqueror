"""TradingView webhook — receives Pine Script alerts and injects signals.

TradingView fires HTTP POST alerts from *their* servers, so this works even
when your laptop is off.  The alert payload must be JSON with the fields
below (set in TradingView's alert message box):

    {
      "symbol":     "{{ticker}}",
      "direction":  "BUY",
      "entry":      {{close}},
      "sl":         <stop price>,
      "tp1":        <target 1>,
      "tp2":        <target 2>,
      "rr":         1.3,
      "confidence": 70,
      "reason":     "MACD bullish divergence"
    }

Optional field: "secret" — set WEBHOOK_SECRET env var to require it.

Railway URL pattern:
    POST https://<your-railway-app>.railway.app/webhook/tradingview
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory as async_sessionmaker
from app.models.signal import Signal
from app.models.strategy import Strategy

router = APIRouter(prefix="/webhook", tags=["webhook"])

_WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

# Symbol → strategy name mapping (must match DB)
_SYMBOL_STRATEGY: dict[str, str] = {
    "ZECUSDT":  "zec_macd_reversal",
    "BTCUSDT":  "btc_macd_reversal",
    "BCHUSDT":  "bch_macd_reversal",
    "ETHUSDT":  "eth_macd_reversal",
}


class TVAlert(BaseModel):
    symbol:     str
    direction:  str          # "BUY" or "SELL"
    entry:      float
    sl:         float
    tp1:        float
    tp2:        float | None = None
    rr:         float        = 1.0
    confidence: float        = 65.0
    reason:     str          = "TradingView alert"
    secret:     str          = ""

    @field_validator("direction")
    @classmethod
    def normalise_direction(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in ("BUY", "SELL"):
            raise ValueError(f"direction must be BUY or SELL, got '{v}'")
        return v

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.upper().strip()


def _to_dec(value: float, places: int = 8) -> Decimal:
    return Decimal(str(round(value, places)))


@router.post("/tradingview")
async def tradingview_webhook(alert: TVAlert) -> dict[str, Any]:
    """Accept a TradingView Pine Script alert and create a trade signal."""

    # ── Auth ─────────────────────────────────────────────────────────────
    if _WEBHOOK_SECRET and alert.secret != _WEBHOOK_SECRET:
        logger.warning("Webhook: rejected request — bad secret for {}", alert.symbol)
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    # ── Validate symbol ──────────────────────────────────────────────────
    strategy_name = _SYMBOL_STRATEGY.get(alert.symbol)
    if not strategy_name:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown symbol '{alert.symbol}'. Supported: {list(_SYMBOL_STRATEGY)}"
        )

    logger.info(
        "Webhook: {} {} @ {:.4f}  SL={:.4f}  TP1={:.4f}  conf={:.0f}%  strategy={}",
        alert.symbol, alert.direction, alert.entry, alert.sl, alert.tp1,
        alert.confidence, strategy_name,
    )

    async with async_sessionmaker() as session:
        # ── Resolve strategy_id ──────────────────────────────────────────
        result = await session.execute(
            select(Strategy).where(Strategy.name == strategy_name)
        )
        strategy_row = result.scalar_one_or_none()
        if strategy_row is None:
            raise HTTPException(
                status_code=500,
                detail=f"Strategy '{strategy_name}' not found in DB — has the bot bootstrapped?"
            )

        # ── Build and store signal ───────────────────────────────────────
        tp2 = alert.tp2 if alert.tp2 is not None else alert.tp1  # fallback
        signal = Signal(
            strategy_id   = strategy_row.id,
            symbol        = alert.symbol,
            timeframe     = "H1",
            direction     = alert.direction,
            entry_price   = _to_dec(alert.entry),
            stop_loss     = _to_dec(alert.sl),
            take_profit_1 = _to_dec(alert.tp1),
            take_profit_2 = _to_dec(tp2),
            risk_reward   = _to_dec(alert.rr, 2),
            confidence    = _to_dec(alert.confidence, 2),
            reasoning     = f"[TradingView] {alert.reason}",
            status        = "active",
            expires_at    = datetime.now(timezone.utc) + timedelta(hours=8),
        )
        session.add(signal)
        await session.commit()
        await session.refresh(signal)

        logger.info("Webhook: signal #{} stored for {} {}", signal.id, alert.symbol, alert.direction)

    return {
        "ok":          True,
        "signal_id":   signal.id,
        "symbol":      alert.symbol,
        "direction":   alert.direction,
        "strategy":    strategy_name,
        "entry":       alert.entry,
    }
