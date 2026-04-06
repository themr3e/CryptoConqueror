"""Telegram notification service for trading signals and system alerts.

Sends formatted HTML messages via Telegram Bot API with built-in retry
logic and rate limiting. Designed as fire-and-forget -- failures are
logged but never raised to callers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
from loguru import logger


class TelegramNotifier:
    """Send HTML-formatted Telegram messages with retry and rate limiting."""

    TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
    MAX_RETRIES = 3
    RATE_LIMIT_SECONDS = 1.0

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self._lock = asyncio.Lock()
        self._last_sent: datetime | None = None

    async def _rate_limit(self) -> None:
        """Enforce 1 message/second rate limit."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            if self._last_sent is not None:
                elapsed = (now - self._last_sent).total_seconds()
                if elapsed < self.RATE_LIMIT_SECONDS:
                    await asyncio.sleep(self.RATE_LIMIT_SECONDS - elapsed)
            self._last_sent = datetime.now(timezone.utc)

    async def _send_message(self, text: str) -> None:
        """Send an HTML-formatted message via Telegram Bot API."""
        if not self.enabled:
            return

        await self._rate_limit()

        url = self.TELEGRAM_API.format(token=self.bot_token)
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 429:
                        logger.warning("Telegram rate limited, skipping")
                        return
                    resp.raise_for_status()
                    return
            except httpx.HTTPStatusError as exc:
                logger.warning("Telegram HTTP error {}: {}", exc.response.status_code, exc)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as exc:
                logger.warning("Telegram send error: {}", exc)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)

    async def notify_signal(self, signal, strategy_name: str = "Unknown") -> None:
        """Send a trade signal alert."""
        if not self.enabled:
            return
        try:
            direction_emoji = "📈" if signal.direction == "BUY" else "📉"
            text = (
                f"{direction_emoji} <b>New Signal: {signal.direction} {signal.symbol}</b>\n\n"
                f"<b>Strategy:</b> {strategy_name}\n"
                f"<b>Entry:</b> {float(signal.entry_price):.2f}\n"
                f"<b>Stop Loss:</b> {float(signal.stop_loss):.2f}\n"
                f"<b>TP1:</b> {float(signal.take_profit_1):.2f}\n"
                f"<b>TP2:</b> {float(signal.take_profit_2):.2f}\n"
                f"<b>R:R:</b> {float(signal.risk_reward):.2f}\n"
                f"<b>Confidence:</b> {float(signal.confidence):.0f}%\n"
            )
            await self._send_message(text)
        except Exception:
            logger.exception("notify_signal failed")

    async def notify_outcome(self, signal, outcome) -> None:
        """Send a trade outcome notification."""
        if not self.enabled:
            return
        try:
            result_map = {
                "tp1_hit": "✅ TP1 Hit",
                "tp2_hit": "✅✅ TP2 Hit",
                "sl_hit": "❌ Stop Loss Hit",
                "expired": "⏰ Expired",
            }
            label = result_map.get(outcome.result, outcome.result)
            pnl_val = float(outcome.pnl_usdt) if outcome.pnl_usdt is not None else 0.0
            pnl_sign = "+" if pnl_val >= 0 else ""
            text = (
                f"{label}\n\n"
                f"<b>Symbol:</b> {signal.symbol}\n"
                f"<b>Direction:</b> {signal.direction}\n"
                f"<b>Entry:</b> {float(signal.entry_price):.4f}\n"
                f"<b>Exit:</b> {float(outcome.exit_price):.4f}\n"
                f"<b>PnL:</b> {pnl_sign}{pnl_val:.2f} USDT\n"
            )
            await self._send_message(text)
        except Exception:
            logger.exception("notify_outcome failed")

    async def notify_degradation(
        self, strategy_name: str, reason: str, is_recovery: bool = False
    ) -> None:
        """Send a strategy degradation or recovery alert."""
        if not self.enabled:
            return
        try:
            emoji = "✅" if is_recovery else "⚠️"
            label = "Recovered" if is_recovery else "Degraded"
            text = (
                f"{emoji} <b>Strategy {label}: {strategy_name}</b>\n\n"
                f"{reason}"
            )
            await self._send_message(text)
        except Exception:
            logger.exception("notify_degradation failed")

    async def notify_circuit_breaker(self, active: bool, reason: str = "") -> None:
        """Send a circuit breaker status notification."""
        if not self.enabled:
            return
        try:
            emoji = "🔴" if active else "🟢"
            status = "ACTIVATED" if active else "RESET"
            text = f"{emoji} <b>Circuit Breaker {status}</b>\n\n{reason}"
            await self._send_message(text)
        except Exception:
            logger.exception("notify_circuit_breaker failed")

    async def notify_system_alert(self, title: str, body: str) -> None:
        """Send a system alert notification."""
        if not self.enabled:
            return
        try:
            text = f"🚨 <b>{title}</b>\n\n{body}"
            await self._send_message(text)
        except Exception:
            logger.exception("notify_system_alert failed")

    async def notify_health_digest(self, stats: dict) -> None:
        """Send the daily health digest."""
        if not self.enabled:
            return
        try:
            failures = stats.get("job_failures", {})
            failing_jobs = [k for k, v in failures.items() if v > 0]
            failure_str = ", ".join(failing_jobs) if failing_jobs else "None"

            text = (
                f"📊 <b>Daily Health Digest</b>\n\n"
                f"<b>Active Signals:</b> {stats.get('active_signals', 0)}\n"
                f"<b>Outcomes Today:</b> {stats.get('outcomes_today', 0)}\n\n"
                f"<b>Candles:</b>\n"
                f"  M15: {stats.get('candles_m15', 0)}\n"
                f"  H1: {stats.get('candles_h1', 0)}\n"
                f"  H4: {stats.get('candles_h4', 0)}\n"
                f"  D1: {stats.get('candles_d1', 0)}\n\n"
                f"<b>Failing Jobs:</b> {failure_str}"
            )
            await self._send_message(text)
        except Exception:
            logger.exception("notify_health_digest failed")
