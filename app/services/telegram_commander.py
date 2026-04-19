"""Telegram two-way command interface for the trading bot.

Polls Telegram for incoming messages and responds to commands.
Only processes messages from the authorized TELEGRAM_CHAT_ID.

Supported commands:
    /help       — list all commands
    /status     — active positions + today's P&L + win rate summary
    /positions  — detailed list of all open trades
    /pnl        — today's P&L breakdown
    /winrate    — 7d and 30d win rates per strategy
    /balance    — Binance account USDT balance
    /pause      — pause the Claude agent (no new trades)
    /resume     — resume the Claude agent
    /top        — top 3 performing symbols this week
    /worst      — worst 3 performing symbols this week
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.outcome import Outcome
from app.models.signal import Signal


# Module-level pause flag — shared across jobs
_agent_paused: bool = False
_last_update_id: int = 0  # tracks Telegram message offset


def is_agent_paused() -> bool:
    """Return True when the operator has paused the agent via Telegram."""
    return _agent_paused


class TelegramCommander:
    """Polls Telegram for commands and executes them against the DB."""

    GET_UPDATES_URL = "https://api.telegram.org/bot{token}/getUpdates"
    SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self) -> None:
        settings = get_settings()
        self._token = settings.telegram_bot_token or ""
        self._chat_id = str(settings.telegram_chat_id or "")
        self._enabled = bool(self._token and self._chat_id)

    # ── Public entry point ────────────────────────────────────────────────────

    async def poll_and_handle(self, session: AsyncSession) -> None:
        """Fetch new Telegram updates and handle any commands. Called every 15s."""
        global _last_update_id

        if not self._enabled:
            return

        updates = await self._get_updates(_last_update_id + 1)
        for update in updates:
            update_id = update.get("update_id", 0)
            _last_update_id = max(_last_update_id, update_id)

            message = update.get("message") or update.get("edited_message")
            if not message:
                continue

            # Only process messages from authorized chat
            chat_id = str(message.get("chat", {}).get("id", ""))
            if chat_id != self._chat_id:
                logger.warning("TelegramCommander: ignored message from unknown chat {}", chat_id)
                continue

            text = (message.get("text") or "").strip()
            if not text.startswith("/"):
                continue

            command = text.split()[0].lower().split("@")[0]  # strip @botname suffix
            logger.info("TelegramCommander: received command '{}' from chat {}", command, chat_id)

            try:
                response = await self._dispatch(command, session)
            except Exception as exc:
                logger.opt(exception=True).error("TelegramCommander: error handling {}", command)
                response = f"⚠️ Error processing <b>{command}</b>: {exc}"

            await self._send(response)

    # ── Command dispatcher ────────────────────────────────────────────────────

    async def _dispatch(self, command: str, session: AsyncSession) -> str:
        handlers = {
            "/help":      self._cmd_help,
            "/status":    self._cmd_status,
            "/positions": self._cmd_positions,
            "/pnl":       self._cmd_pnl,
            "/trades":    self._cmd_trades,
            "/winrate":   self._cmd_winrate,
            "/balance":   self._cmd_balance,
            "/pause":     self._cmd_pause,
            "/resume":    self._cmd_resume,
            "/top":       self._cmd_top,
            "/worst":     self._cmd_worst,
        }
        handler = handlers.get(command)
        if handler is None:
            return f"Unknown command: <b>{command}</b>\nSend /help for the full list."
        return await handler(session)

    # ── Command implementations ───────────────────────────────────────────────

    async def _cmd_help(self, session: AsyncSession) -> str:
        return (
            "🤖 <b>Zafir Trading Bot — Commands</b>\n\n"
            "/status    — open positions + today's P&amp;L\n"
            "/positions — detailed open trades\n"
            "/pnl       — today's P&amp;L breakdown\n"
            "/trades    — full trade journal with Claude's reasoning\n"
            "/winrate   — 7d and 30d win rates\n"
            "/balance   — Binance account balance\n"
            "/top       — best symbols this week\n"
            "/worst     — worst symbols this week\n"
            "/pause     — stop taking new trades\n"
            "/resume    — resume taking trades\n"
            "/help      — this message"
        )

    async def _cmd_status(self, session: AsyncSession) -> str:
        global _agent_paused

        # Open positions
        result = await session.execute(
            select(Signal).where(Signal.status == "active")
        )
        open_signals = result.scalars().all()

        # Today's P&L
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        pnl_result = await session.execute(
            select(func.coalesce(func.sum(Outcome.pnl_usdt), 0)).where(
                Outcome.created_at >= today
            )
        )
        today_pnl = float(pnl_result.scalar_one())

        # Today's trade count
        trades_today = await session.execute(
            select(func.count(Outcome.id)).where(Outcome.created_at >= today)
        )
        trade_count = int(trades_today.scalar_one())

        # 7d win rate
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        week_result = await session.execute(
            select(Outcome).where(Outcome.created_at >= week_ago)
        )
        week_outcomes = week_result.scalars().all()
        wins_7d = sum(1 for o in week_outcomes if o.result in ("tp1_hit", "tp2_hit"))
        total_7d = len(week_outcomes)
        wr_7d = f"{wins_7d}/{total_7d} ({wins_7d/total_7d*100:.0f}%)" if total_7d else "no data"

        agent_status = "⏸ PAUSED" if _agent_paused else "▶️ RUNNING"

        lines = [
            f"📊 <b>Bot Status — {agent_status}</b>",
            "",
            f"<b>Open positions:</b> {len(open_signals)}",
            f"<b>Today's P&amp;L:</b> ${today_pnl:+.2f} ({trade_count} trades)",
            f"<b>Win rate (7d):</b> {wr_7d}",
        ]
        if open_signals:
            lines.append("")
            lines.append("<b>Open trades:</b>")
            for s in open_signals[:5]:
                lines.append(
                    f"  {'📈' if s.direction == 'BUY' else '📉'} {s.symbol} "
                    f"@ {float(s.entry_price):.4f} | SL {float(s.stop_loss):.4f}"
                )

        return "\n".join(lines)

    async def _cmd_positions(self, session: AsyncSession) -> str:
        result = await session.execute(
            select(Signal).where(Signal.status == "active").order_by(Signal.created_at.desc())
        )
        signals = result.scalars().all()

        if not signals:
            return "📭 <b>No open positions.</b>"

        lines = [f"📋 <b>Open Positions ({len(signals)})</b>", ""]
        for s in signals:
            age_mins = int((datetime.now(timezone.utc) - s.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 60)
            lines.append(
                f"{'📈' if s.direction == 'BUY' else '📉'} <b>{s.symbol}</b> — {s.direction}\n"
                f"   Entry: {float(s.entry_price):.4f}\n"
                f"   SL: {float(s.stop_loss):.4f} | TP: {float(s.take_profit_1):.4f}\n"
                f"   RR: {float(s.risk_reward):.1f} | Conf: {float(s.confidence):.0f}%\n"
                f"   Open {age_mins}m ago"
            )
        return "\n".join(lines)

    async def _cmd_pnl(self, session: AsyncSession) -> str:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        result = await session.execute(
            select(Outcome, Signal.symbol, Signal.direction)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.created_at >= today)
            .order_by(Outcome.created_at.desc())
        )
        rows = result.all()

        if not rows:
            return "📭 <b>No closed trades today.</b>"

        total = sum(float(r[0].pnl_usdt or 0) for r in rows)
        wins = sum(1 for r in rows if r[0].result in ("tp1_hit", "tp2_hit"))

        lines = [f"💰 <b>Today's P&amp;L — ${total:+.2f}</b>", f"{wins}/{len(rows)} winners", ""]
        for outcome, symbol, direction in rows[:10]:
            emoji = "✅" if outcome.result in ("tp1_hit", "tp2_hit") else "❌"
            pnl = float(outcome.pnl_usdt or 0)
            lines.append(f"{emoji} {symbol} {direction} → ${pnl:+.2f} ({outcome.result})")

        return "\n".join(lines)

    async def _cmd_trades(self, session: AsyncSession) -> str:
        """Full trade journal for today — entry, SL, TP, result, Claude's reasoning."""
        from app.models.claude_decision import ClaudeDecision

        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        # Closed trades today
        closed_result = await session.execute(
            select(Outcome, Signal)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.created_at >= today)
            .order_by(Outcome.created_at.desc())
        )
        closed_rows = closed_result.all()

        # Open trades
        open_result = await session.execute(
            select(Signal).where(Signal.status == "active").order_by(Signal.created_at.desc())
        )
        open_sigs = open_result.scalars().all()

        if not closed_rows and not open_sigs:
            return "📭 <b>No trades today yet.</b>"

        total_pnl = sum(float(r[0].pnl_usdt or 0) for r in closed_rows)
        wins = sum(1 for r in closed_rows if r[0].result in ("tp1_hit", "tp2_hit"))
        lines = [
            f"📋 <b>Today's Trade Journal</b>",
            f"{wins}/{len(closed_rows)} closed | ${total_pnl:+.2f} P&amp;L",
        ]

        if closed_rows:
            lines.append("")
            lines.append("<b>— Closed —</b>")
            for outcome, signal in closed_rows[:8]:
                result_emoji = "✅" if outcome.result in ("tp1_hit", "tp2_hit") else "❌"
                pnl = float(outcome.pnl_usdt or 0)
                closed_time = outcome.created_at.strftime("%H:%M")

                # Find Claude's reasoning for this trade
                dec_result = await session.execute(
                    select(ClaudeDecision)
                    .where(
                        ClaudeDecision.symbol == signal.symbol,
                        ClaudeDecision.action.in_(["open_long", "open_short"]),
                        ClaudeDecision.created_at >= signal.created_at - timedelta(minutes=5),
                        ClaudeDecision.created_at <= signal.created_at + timedelta(minutes=5),
                    )
                    .order_by(ClaudeDecision.created_at.desc())
                    .limit(1)
                )
                decision = dec_result.scalar_one_or_none()
                reasoning = decision.reasoning[:100] if decision else "—"

                lines.append(
                    f"\n{result_emoji} <b>{signal.symbol}</b> {signal.direction} [{closed_time} UTC]\n"
                    f"   Entry: {float(signal.entry_price):.4f} | SL: {float(signal.stop_loss):.4f} | TP: {float(signal.take_profit_1):.4f}\n"
                    f"   Result: {outcome.result} | P&amp;L: ${pnl:+.2f}\n"
                    f"   Why: {reasoning}"
                )

        if open_sigs:
            lines.append("")
            lines.append("<b>— Still Open —</b>")
            for s in open_sigs:
                age_mins = int((datetime.now(timezone.utc) - s.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 60)
                dec_result = await session.execute(
                    select(ClaudeDecision)
                    .where(
                        ClaudeDecision.symbol == s.symbol,
                        ClaudeDecision.action.in_(["open_long", "open_short"]),
                        ClaudeDecision.created_at >= s.created_at - timedelta(minutes=5),
                        ClaudeDecision.created_at <= s.created_at + timedelta(minutes=5),
                    )
                    .order_by(ClaudeDecision.created_at.desc())
                    .limit(1)
                )
                decision = dec_result.scalar_one_or_none()
                reasoning = decision.reasoning[:100] if decision else "—"

                lines.append(
                    f"\n⏳ <b>{s.symbol}</b> {s.direction} [open {age_mins}m]\n"
                    f"   Entry: {float(s.entry_price):.4f} | SL: {float(s.stop_loss):.4f} | TP: {float(s.take_profit_1):.4f}\n"
                    f"   Why: {reasoning}"
                )

        return "\n".join(lines)

    async def _cmd_winrate(self, session: AsyncSession) -> str:
        lines = ["📈 <b>Win Rate Summary</b>", ""]
        for label, days in [("7 days", 7), ("30 days", 30)]:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            result = await session.execute(
                select(Outcome).where(Outcome.created_at >= cutoff)
            )
            outcomes = result.scalars().all()
            if not outcomes:
                lines.append(f"<b>{label}:</b> no data")
                continue
            wins = sum(1 for o in outcomes if o.result in ("tp1_hit", "tp2_hit"))
            total = len(outcomes)
            pnl = sum(float(o.pnl_usdt or 0) for o in outcomes)
            lines.append(
                f"<b>{label}:</b> {wins}/{total} ({wins/total*100:.0f}%) | P&amp;L ${pnl:+.2f}"
            )
        return "\n".join(lines)

    async def _cmd_balance(self, session: AsyncSession) -> str:
        settings = get_settings()
        if not settings.binance_order_execution_enabled:
            return "⚠️ Binance API keys not configured."
        try:
            from app.services.binance_executor import BinanceExecutor
            executor = BinanceExecutor()
            balance = await executor.get_account_balance()
            if balance is None:
                return "⚠️ Could not fetch balance from Binance."
            env = "TESTNET" if settings.binance_testnet else "MAINNET"
            return f"💳 <b>Binance Balance ({env})</b>\n\n<b>Available USDT:</b> ${float(balance):,.2f}"
        except Exception as exc:
            return f"⚠️ Balance fetch failed: {exc}"

    async def _cmd_pause(self, session: AsyncSession) -> str:
        global _agent_paused
        _agent_paused = True
        logger.warning("TelegramCommander: Claude agent PAUSED by operator")
        return "⏸ <b>Bot paused.</b>\n\nNo new trades will be opened until you send /resume."

    async def _cmd_resume(self, session: AsyncSession) -> str:
        global _agent_paused
        _agent_paused = False
        logger.info("TelegramCommander: Claude agent RESUMED by operator")
        return "▶️ <b>Bot resumed.</b>\n\nLooking for trades again."

    async def _cmd_top(self, session: AsyncSession) -> str:
        return await self._cmd_symbol_perf(session, best=True)

    async def _cmd_worst(self, session: AsyncSession) -> str:
        return await self._cmd_symbol_perf(session, best=False)

    async def _cmd_symbol_perf(self, session: AsyncSession, best: bool) -> str:
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        result = await session.execute(
            select(Signal.symbol, Outcome.result, Outcome.pnl_usdt)
            .join(Outcome, Outcome.signal_id == Signal.id)
            .where(Outcome.created_at >= week_ago)
        )
        rows = result.all()

        if not rows:
            return "📭 No closed trades this week."

        symbol_stats: dict[str, dict] = {}
        for symbol, outcome_result, pnl in rows:
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {"pnl": 0.0, "wins": 0, "total": 0}
            symbol_stats[symbol]["pnl"] += float(pnl or 0)
            symbol_stats[symbol]["total"] += 1
            if outcome_result in ("tp1_hit", "tp2_hit"):
                symbol_stats[symbol]["wins"] += 1

        ranked = sorted(symbol_stats.items(), key=lambda x: x[1]["pnl"], reverse=best)[:3]
        label = "🏆 Top" if best else "📉 Worst"
        lines = [f"{label} <b>Symbols This Week</b>", ""]
        for symbol, stats in ranked:
            wr = stats["wins"] / stats["total"] * 100 if stats["total"] else 0
            lines.append(
                f"<b>{symbol}</b>: ${stats['pnl']:+.2f} | "
                f"{stats['wins']}/{stats['total']} wins ({wr:.0f}%)"
            )
        return "\n".join(lines)

    # ── Telegram API helpers ──────────────────────────────────────────────────

    async def _get_updates(self, offset: int) -> list[dict]:
        if not self._enabled:
            return []
        try:
            url = self.GET_UPDATES_URL.format(token=self._token)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params={
                    "offset": offset,
                    "limit": 10,
                    "timeout": 0,
                })
                if resp.status_code == 200:
                    return resp.json().get("result", [])
        except Exception as exc:
            logger.debug("TelegramCommander: getUpdates failed — {}", exc)
        return []

    async def _send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            url = self.SEND_URL.format(token=self._token)
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(url, json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                })
        except Exception as exc:
            logger.warning("TelegramCommander: send failed — {}", exc)
