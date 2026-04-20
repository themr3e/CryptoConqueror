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

Safety gates (applied before any Binance order):
  1. Minimum confidence (WEBHOOK_MIN_CONFIDENCE env var, default 65)
  2. Daily loss limit — refuses new trades if today's P&L is below limit
  3. Duplicate position block — one active signal per symbol maximum
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, field_validator
from sqlalchemy import func, select

from app.config import get_settings
from app.database import async_session_factory as async_sessionmaker
from app.models.outcome import Outcome
from app.models.signal import Signal
from app.models.strategy import Strategy
from app.services.binance_executor import BinanceExecutor
from app.services.telegram_notifier import TelegramNotifier

router = APIRouter(prefix="/webhook", tags=["webhook"])

_WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
_MIN_CONFIDENCE = float(os.getenv("WEBHOOK_MIN_CONFIDENCE", "65"))

# Symbol → strategy name mapping (must match DB)
_SYMBOL_STRATEGY: dict[str, str] = {
    "ZECUSDT":  "zec_macd_reversal",
    "BTCUSDT":  "btc_macd_reversal",
    "BCHUSDT":  "bch_macd_reversal",
    "ETHUSDT":  "eth_macd_reversal",
}


class TVAlert(BaseModel):
    symbol:     str
    direction:  str
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


async def _get_today_pnl(session) -> float:
    """Sum of all Outcome.pnl_usdt records created since midnight UTC today."""
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await session.execute(
        select(func.sum(Outcome.pnl_usdt)).where(Outcome.created_at >= today)
    )
    total = result.scalar_one_or_none()
    return float(total or 0)


@router.post("/tradingview")
async def tradingview_webhook(alert: TVAlert) -> dict[str, Any]:
    """Accept a TradingView Pine Script alert and create a trade signal."""
    settings = get_settings()
    notifier = TelegramNotifier(
        bot_token=settings.telegram_bot_token or "",
        chat_id=settings.telegram_chat_id or "",
    )

    # ── Auth ──────────────────────────────────────────────────────────────
    if _WEBHOOK_SECRET and alert.secret != _WEBHOOK_SECRET:
        logger.warning("Webhook: rejected — bad secret for {}", alert.symbol)
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    # ── Validate symbol ───────────────────────────────────────────────────
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

        # ── Gate 1: Minimum confidence ────────────────────────────────────
        if alert.confidence < _MIN_CONFIDENCE:
            msg = (
                f"⚠️ <b>Webhook rejected</b> — low confidence\n"
                f"{alert.symbol} {alert.direction} conf={alert.confidence:.0f}% "
                f"(min={_MIN_CONFIDENCE:.0f}%)"
            )
            logger.warning("Webhook: {} — confidence {:.0f}% below threshold {:.0f}%",
                           alert.symbol, alert.confidence, _MIN_CONFIDENCE)
            await notifier._send_message(msg)
            return {
                "ok": False,
                "rejected": True,
                "reason": f"confidence {alert.confidence:.0f}% < minimum {_MIN_CONFIDENCE:.0f}%",
            }

        # ── Gate 2: Daily loss limit ──────────────────────────────────────
        daily_pnl = await _get_today_pnl(session)
        daily_loss_limit = float(settings.account_balance) * float(settings.claude_agent_daily_loss_limit)
        if daily_pnl <= -daily_loss_limit:
            msg = (
                f"🚨 <b>Webhook blocked — daily loss limit hit</b>\n"
                f"Today's P&L: ${daily_pnl:+.2f}  |  Limit: -${daily_loss_limit:.2f}\n"
                f"Signal {alert.symbol} {alert.direction} rejected. No new trades today."
            )
            logger.warning("Webhook: {} — daily loss limit hit (${:.2f})", alert.symbol, daily_pnl)
            await notifier._send_message(msg)
            return {
                "ok": False,
                "rejected": True,
                "reason": f"daily loss limit hit (today P&L ${daily_pnl:+.2f})",
            }

        # ── Gate 3: No duplicate position ─────────────────────────────────
        active_result = await session.execute(
            select(Signal).where(
                Signal.symbol == alert.symbol,
                Signal.status == "active",
            ).limit(1)
        )
        existing = active_result.scalar_one_or_none()
        if existing is not None:
            logger.info(
                "Webhook: {} — already have active signal #{} {} — blocking duplicate",
                alert.symbol, existing.id, existing.direction,
            )
            return {
                "ok": False,
                "rejected": True,
                "reason": f"active signal #{existing.id} already open for {alert.symbol}",
            }

        # ── Resolve strategy_id ───────────────────────────────────────────
        result = await session.execute(
            select(Strategy).where(Strategy.name == strategy_name)
        )
        strategy_row = result.scalar_one_or_none()
        if strategy_row is None:
            raise HTTPException(
                status_code=500,
                detail=f"Strategy '{strategy_name}' not found in DB — has the bot bootstrapped?"
            )

        # ── Store signal ──────────────────────────────────────────────────
        tp2 = alert.tp2 if alert.tp2 is not None else alert.tp1
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

    # ── Execute on Binance ────────────────────────────────────────────────
    execution_result = None
    if settings.binance_order_execution_enabled:
        executor = BinanceExecutor()
        async with async_sessionmaker() as exec_session:
            execution_result = await executor.execute_signal(exec_session, signal)
            if execution_result.success:
                logger.info(
                    "Webhook: Binance order placed for signal #{} {} {}",
                    signal.id, alert.symbol, alert.direction,
                )
                await notifier._send_message(
                    f"✅ <b>TradingView signal executed</b>\n"
                    f"{alert.symbol} {alert.direction} @ {alert.entry:.4f}\n"
                    f"SL={alert.sl:.4f}  TP={alert.tp1:.4f}  conf={alert.confidence:.0f}%\n"
                    f"Signal #{signal.id}"
                )
            else:
                logger.warning(
                    "Webhook: Binance execution failed for signal #{} — {}",
                    signal.id, execution_result.error_message,
                )
                await notifier._send_message(
                    f"❌ <b>TradingView signal stored but Binance FAILED</b>\n"
                    f"Signal #{signal.id} {alert.symbol} {alert.direction}\n"
                    f"Error: {execution_result.error_message}"
                )
    else:
        logger.info(
            "Webhook: signal #{} stored — Binance execution disabled",
            signal.id,
        )

    return {
        "ok":               True,
        "signal_id":        signal.id,
        "symbol":           alert.symbol,
        "direction":        alert.direction,
        "strategy":         strategy_name,
        "entry":            alert.entry,
        "confidence":       alert.confidence,
        "binance_executed": execution_result.success if execution_result else False,
        "binance_error":    execution_result.error_message if execution_result and not execution_result.success else None,
    }
