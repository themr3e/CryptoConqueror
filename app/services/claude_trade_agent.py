"""Claude autonomous trading agent.

Uses the Anthropic API to analyze market conditions and make trading decisions
for Binance Futures. Claude acts as the decision maker — the existing
BinanceExecutor handles order placement.

Risk guardrails (enforced in code, never overridable by Claude):
  - Max 5% account risk per trade (configurable via CLAUDE_AGENT_RISK_PCT)
  - Max 10% daily loss limit (configurable via CLAUDE_AGENT_DAILY_LOSS_LIMIT)
  - All decisions logged to claude_decisions table
  - Every action reported to Telegram
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import anthropic
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.claude_decision import ClaudeDecision
from app.models.outcome import Outcome
from app.models.signal import Signal
from app.services.telegram_notifier import TelegramNotifier


_SYSTEM_PROMPT = """You are an expert Binance Futures trader managing a live trading account.
Your goal is to generate consistent profit while strictly managing risk.

You will receive market data for MULTIPLE symbols at once.

You must respond with a JSON ARRAY ONLY — one object per symbol, no explanation outside the JSON.

Response format:
[
  {
    "symbol": "BTCUSDT",
    "action": "open_long" | "open_short" | "close" | "hold",
    "reasoning": "Brief explanation (1-2 sentences)",
    "confidence": 0-100,
    "entry_price": <number or null>,
    "stop_loss": <number or null>,
    "take_profit": <number or null>
  },
  ...
]

Rules:
- Return exactly one object per symbol provided — same order as input
- Use "hold" when conditions are unclear or risky
- Always set stop_loss and take_profit for open_long/open_short
- stop_loss must be at least 1.5% from entry
- take_profit must give minimum 1.5:1 risk/reward ratio
- Consider H4 and H1 trends before entering
- Avoid trading against the dominant trend
- If daily loss limit is near, prefer "hold" for all symbols

"""


class ClaudeTradeAgent:
    """Autonomous trading agent powered by Claude."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._settings = settings
        self._notifier = TelegramNotifier(
            bot_token=settings.telegram_bot_token or "",
            chat_id=settings.telegram_chat_id or "",
        )

    async def run(self, session: AsyncSession, symbol: str) -> ClaudeDecision:
        """Run one decision cycle for a symbol. Logs and optionally executes."""
        settings = self._settings

        # ── Guard: daily loss limit ───────────────────────────────────────────
        daily_pnl = await self._get_daily_pnl(session)
        daily_loss_limit = settings.account_balance * settings.claude_agent_daily_loss_limit
        if daily_pnl <= -daily_loss_limit:
            logger.warning(
                "[ClaudeAgent] Daily loss limit hit (${:.2f}) — holding all positions for {}",
                daily_pnl, symbol,
            )
            decision = ClaudeDecision(
                symbol=symbol,
                action="hold",
                reasoning=f"Daily loss limit reached (${abs(daily_pnl):.2f} lost today). No new trades.",
                confidence=100.0,
                daily_pnl_at_decision=daily_pnl,
                executed=False,
            )
            session.add(decision)
            await session.commit()
            return decision

        # ── Sync DB signals with actual Binance positions ─────────────────────
        # Stale "active" signals in the DB cause Claude to incorrectly think
        # a position is open. Cross-check with Binance and close any ghosts.
        if settings.binance_order_execution_enabled:
            try:
                from app.services.binance_executor import BinanceExecutor
                executor = BinanceExecutor()
                actual_size = await executor.get_open_position_size(symbol)
                if actual_size == 0.0:
                    # No real position — close any stale active signals in DB
                    stale_result = await session.execute(
                        select(Signal).where(Signal.symbol == symbol, Signal.status == "active")
                    )
                    stale_signals = stale_result.scalars().all()
                    if stale_signals:
                        for s in stale_signals:
                            s.status = "closed"
                        await session.commit()
                        logger.info(
                            "[ClaudeAgent] Synced {} stale active signal(s) → closed for {} (no Binance position)",
                            len(stale_signals), symbol,
                        )
            except Exception:
                logger.opt(exception=True).warning("[ClaudeAgent] Position sync failed for {} — continuing", symbol)

        # ── Build market context ──────────────────────────────────────────────
        context = await self._build_context(session, symbol, daily_pnl)

        # ── Call Claude ───────────────────────────────────────────────────────
        raw_decision = await self._ask_claude(context)

        # ── Parse response ────────────────────────────────────────────────────
        action = raw_decision.get("action", "hold")
        reasoning = raw_decision.get("reasoning", "No reasoning provided")
        confidence = float(raw_decision.get("confidence", 50))
        entry_price = raw_decision.get("entry_price")
        stop_loss = raw_decision.get("stop_loss")
        take_profit = raw_decision.get("take_profit")

        decision = ClaudeDecision(
            symbol=symbol,
            action=action,
            reasoning=reasoning,
            confidence=confidence,
            entry_price=Decimal(str(entry_price)) if entry_price else None,
            stop_loss=Decimal(str(stop_loss)) if stop_loss else None,
            take_profit=Decimal(str(take_profit)) if take_profit else None,
            daily_pnl_at_decision=daily_pnl,
            executed=False,
        )

        logger.info(
            "[ClaudeAgent] Decision for {}: {} (confidence: {}%) — {}",
            symbol, action, confidence, reasoning[:80],
        )

        # ── Execute if action requires it ─────────────────────────────────────
        if action in ("open_long", "open_short") and entry_price and stop_loss and take_profit:
            executed, error = await self._execute(session, decision, symbol)
            decision.executed = executed
            decision.execution_error = error

        elif action == "close":
            executed, error = await self._close_open_positions(session, symbol)
            decision.executed = executed
            decision.execution_error = error

        session.add(decision)
        await session.commit()

        # ── Notify Telegram ───────────────────────────────────────────────────
        await self._notify(decision)

        return decision

    async def run_batch(
        self, session: AsyncSession, symbols: list[str]
    ) -> list[ClaudeDecision]:
        """Run one decision cycle for all symbols in a single Claude API call.

        This is far more efficient than calling run() per symbol — 1 API call
        instead of N, reducing total wall time from minutes to ~10-30 seconds.
        """
        settings = self._settings
        daily_pnl = await self._get_daily_pnl(session)
        daily_loss_limit = settings.account_balance * settings.claude_agent_daily_loss_limit

        # Position sync for all symbols
        if settings.binance_order_execution_enabled:
            from app.services.binance_executor import BinanceExecutor
            executor = BinanceExecutor()
            for symbol in symbols:
                try:
                    actual_size = await executor.get_open_position_size(symbol)
                    if actual_size == 0.0:
                        stale_result = await session.execute(
                            select(Signal).where(Signal.symbol == symbol, Signal.status == "active")
                        )
                        stale_signals = stale_result.scalars().all()
                        if stale_signals:
                            for s in stale_signals:
                                s.status = "closed"
                            await session.commit()
                            logger.info(
                                "[ClaudeAgent] Synced {} stale signal(s) → closed for {} (no position)",
                                len(stale_signals), symbol,
                            )
                except Exception:
                    logger.opt(exception=True).warning("[ClaudeAgent] Position sync failed for {}", symbol)

        # Daily loss limit check
        if daily_pnl <= -daily_loss_limit:
            logger.warning("[ClaudeAgent] Daily loss limit hit — holding all {} symbols", len(symbols))
            decisions = []
            for symbol in symbols:
                d = ClaudeDecision(
                    symbol=symbol,
                    action="hold",
                    reasoning=f"Daily loss limit reached (${abs(daily_pnl):.2f} lost today).",
                    confidence=100.0,
                    daily_pnl_at_decision=daily_pnl,
                    executed=False,
                )
                session.add(d)
                decisions.append(d)
            await session.commit()
            return decisions

        # Build combined context for all symbols
        context = await self._build_batch_context(session, symbols, daily_pnl)

        # Single Claude API call for all symbols
        raw_decisions = await self._ask_claude_batch(context, symbols)

        decisions = []
        for raw in raw_decisions:
            symbol = raw.get("symbol", "UNKNOWN")
            action = raw.get("action", "hold")
            reasoning = raw.get("reasoning", "No reasoning provided")
            confidence = float(raw.get("confidence", 50))
            entry_price = raw.get("entry_price")
            stop_loss = raw.get("stop_loss")
            take_profit = raw.get("take_profit")

            decision = ClaudeDecision(
                symbol=symbol,
                action=action,
                reasoning=reasoning,
                confidence=confidence,
                entry_price=Decimal(str(entry_price)) if entry_price else None,
                stop_loss=Decimal(str(stop_loss)) if stop_loss else None,
                take_profit=Decimal(str(take_profit)) if take_profit else None,
                daily_pnl_at_decision=daily_pnl,
                executed=False,
            )

            logger.info(
                "[ClaudeAgent] {} → {} ({}%) — {}",
                symbol, action, int(confidence), reasoning[:60],
            )

            if action in ("open_long", "open_short") and entry_price and stop_loss and take_profit:
                executed, error = await self._execute(session, decision, symbol)
                decision.executed = executed
                decision.execution_error = error
            elif action == "close":
                executed, error = await self._close_open_positions(session, symbol)
                decision.executed = executed
                decision.execution_error = error

            session.add(decision)
            decisions.append(decision)

        await session.commit()

        # Notify Telegram with a summary instead of 32 separate messages
        await self._notify_batch_summary(decisions, daily_pnl)

        return decisions

    async def _build_batch_context(
        self, session: AsyncSession, symbols: list[str], daily_pnl: float
    ) -> str:
        """Build combined market context for all symbols."""
        from app.models.candle import Candle

        lines = [
            f"Account Balance: ${self._settings.account_balance:,.2f}",
            f"Today's P&L: ${daily_pnl:+.2f}",
            f"Environment: {'TESTNET' if self._settings.binance_testnet else 'MAINNET'}",
            f"Symbols to analyze: {len(symbols)}",
            "",
        ]

        for symbol in symbols:
            lines.append(f"=== {symbol} ===")

            for tf in ["H1", "H4"]:
                result = await session.execute(
                    select(Candle)
                    .where(Candle.symbol == symbol, Candle.timeframe == tf)
                    .order_by(Candle.timestamp.desc())
                    .limit(5)
                )
                candles = list(reversed(result.scalars().all()))
                if candles:
                    lines.append(f"[{tf}] " + " | ".join(
                        f"{float(c.close):.4f}" for c in candles
                    ) + " (latest close)")

            open_result = await session.execute(
                select(Signal).where(Signal.symbol == symbol, Signal.status == "active")
            )
            open_sigs = open_result.scalars().all()
            if open_sigs:
                s = open_sigs[0]
                lines.append(f"OPEN: {s.direction} @ {float(s.entry_price):.4f} SL={float(s.stop_loss):.4f} TP={float(s.take_profit_1):.4f}")
            else:
                lines.append("OPEN: none")

            # Recent win rate for this symbol (last 10 outcomes)
            recent_result = await session.execute(
                select(Outcome)
                .join(Signal, Outcome.signal_id == Signal.id)
                .where(Signal.symbol == symbol)
                .order_by(Outcome.created_at.desc())
                .limit(10)
            )
            recent = recent_result.scalars().all()
            if recent:
                wins = sum(1 for o in recent if o.result in ("tp1_hit", "tp2_hit"))
                total = len(recent)
                total_pnl = sum(float(o.pnl_usdt or 0) for o in recent)
                lines.append(f"HISTORY: {wins}/{total} wins ({wins/total*100:.0f}%) | P&L=${total_pnl:+.2f}")
            else:
                lines.append("HISTORY: no closed trades yet")
            lines.append("")

        return "\n".join(lines)

    async def _ask_claude_batch(self, context: str, symbols: list[str]) -> list[dict]:
        """Send batch context to Claude and parse the JSON array response."""
        try:
            message = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            text = message.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            if isinstance(result, list):
                return result
            # If Claude returned a single object, wrap it
            return [result]
        except Exception:
            logger.opt(exception=True).error("[ClaudeAgent] Batch API call failed — defaulting all to hold")
            return [
                {"symbol": s, "action": "hold", "reasoning": "Claude API error", "confidence": 0}
                for s in symbols
            ]

    async def _notify_batch_summary(
        self, decisions: list[ClaudeDecision], daily_pnl: float
    ) -> None:
        """Send a single Telegram summary for all batch decisions."""
        try:
            env = "TESTNET" if self._settings.binance_testnet else "MAINNET"
            action_counts: dict[str, int] = {}
            trades = []
            for d in decisions:
                action_counts[d.action] = action_counts.get(d.action, 0) + 1
                if d.action in ("open_long", "open_short"):
                    emoji = "📈" if d.action == "open_long" else "📉"
                    status = "✅" if d.executed else "❌"
                    trades.append(
                        f"{emoji} {status} <b>{d.symbol}</b> {d.action.upper()} "
                        f"@ {float(d.entry_price):.4f} | conf={d.confidence:.0f}%"
                    )

            lines = [
                f"🤖 <b>Claude Agent Batch — {env}</b>",
                f"<b>Daily P&L:</b> ${daily_pnl:+.2f}",
                f"<b>Summary:</b> " + ", ".join(f"{k}={v}" for k, v in sorted(action_counts.items())),
            ]
            if trades:
                lines.append("")
                lines.extend(trades)

            await self._notifier._send_message("\n".join(lines))
        except Exception:
            logger.opt(exception=True).warning("[ClaudeAgent] Batch Telegram notify failed")

    async def _build_context(self, session: AsyncSession, symbol: str, daily_pnl: float) -> str:
        """Build market context string to send to Claude."""
        from app.models.candle import Candle

        lines = [f"=== MARKET CONTEXT: {symbol} ==="]
        lines.append(f"Account Balance: ${self._settings.account_balance:,.2f}")
        lines.append(f"Today's P&L: ${daily_pnl:+.2f}")
        lines.append(f"Environment: {'TESTNET' if self._settings.binance_testnet else 'MAINNET'}")
        lines.append("")

        # Candles per timeframe
        for tf in ["H1", "H4", "D1"]:
            result = await session.execute(
                select(Candle)
                .where(Candle.symbol == symbol, Candle.timeframe == tf)
                .order_by(Candle.timestamp.desc())
                .limit(10)
            )
            candles = list(reversed(result.scalars().all()))
            if candles:
                lines.append(f"--- {tf} Candles (last {len(candles)}) ---")
                lines.append("timestamp, open, high, low, close, volume")
                for c in candles:
                    lines.append(
                        f"{c.timestamp.strftime('%Y-%m-%d %H:%M')}, "
                        f"{float(c.open):.2f}, {float(c.high):.2f}, "
                        f"{float(c.low):.2f}, {float(c.close):.2f}, "
                        f"{float(c.volume):.2f}"
                    )
                lines.append("")

        # Open positions
        open_signals_result = await session.execute(
            select(Signal).where(Signal.symbol == symbol, Signal.status == "active")
        )
        open_signals = open_signals_result.scalars().all()
        if open_signals:
            lines.append("--- Open Positions ---")
            for s in open_signals:
                lines.append(
                    f"{s.direction} entry={float(s.entry_price):.2f} "
                    f"sl={float(s.stop_loss):.2f} tp={float(s.take_profit_1):.2f}"
                )
        else:
            lines.append("--- Open Positions: None ---")
        lines.append("")

        # Recent outcomes (last 10 — use pnl_usdt for crypto)
        recent_result = await session.execute(
            select(Outcome)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Signal.symbol == symbol)
            .order_by(Outcome.created_at.desc())
            .limit(10)
        )
        recent = recent_result.scalars().all()
        if recent:
            wins = sum(1 for o in recent if o.result in ("tp1_hit", "tp2_hit"))
            total = len(recent)
            win_rate = wins / total * 100
            total_pnl = sum(float(o.pnl_usdt or 0) for o in recent)
            lines.append(f"--- Recent Outcomes (last {total}) ---")
            lines.append(f"Win rate: {wins}/{total} ({win_rate:.0f}%) | Total P&L: ${total_pnl:+.2f}")
            for o in recent:
                lines.append(f"  {o.result} pnl=${float(o.pnl_usdt or 0):+.2f}")
        else:
            lines.append("--- Recent Outcomes: none yet ---")
        lines.append("")

        return "\n".join(lines)

    async def _ask_claude(self, context: str) -> dict:
        """Send context to Claude and parse the JSON response."""
        try:
            message = await self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            text = message.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception:
            logger.opt(exception=True).error("[ClaudeAgent] Failed to get/parse Claude response")
            return {"action": "hold", "reasoning": "Claude API error — defaulting to hold", "confidence": 0}

    async def _execute(
        self, session: AsyncSession, decision: ClaudeDecision, symbol: str
    ) -> tuple[bool, str | None]:
        """Execute an open_long or open_short via the signal pipeline."""
        try:
            from app.models.strategy import Strategy
            from app.models.signal import Signal as SignalModel
            from sqlalchemy import select as sa_select
            import decimal

            settings = self._settings

            # Get or create a "claude_agent" strategy entry
            strat_result = await session.execute(
                sa_select(Strategy).where(Strategy.name == "claude_agent")
            )
            strategy = strat_result.scalar_one_or_none()
            if not strategy:
                strategy = Strategy(
                    name="claude_agent",
                    is_active=True,
                    asset_class="crypto_futures",
                    symbols='["BTCUSDT","ETHUSDT"]',
                )
                session.add(strategy)
                await session.flush()

            entry = float(decision.entry_price)
            sl = float(decision.stop_loss)
            tp = float(decision.take_profit)
            direction = "BUY" if decision.action == "open_long" else "SELL"

            risk_distance = abs(entry - sl)
            rr = abs(tp - entry) / risk_distance if risk_distance > 0 else 0
            risk_amount = settings.account_balance * settings.claude_agent_risk_pct
            position_size = risk_amount / risk_distance if risk_distance > 0 else 0
            decision.position_size = Decimal(str(round(position_size, 6)))

            signal = SignalModel(
                strategy_id=strategy.id,
                symbol=symbol,
                timeframe="H1",
                direction=direction,
                entry_price=Decimal(str(entry)),
                stop_loss=Decimal(str(sl)),
                take_profit_1=Decimal(str(tp)),
                take_profit_2=Decimal(str(tp)),
                risk_reward=Decimal(str(round(rr, 2))),
                confidence=Decimal(str(decision.confidence or 50)),
                reasoning=decision.reasoning,
                status="active",
            )
            session.add(signal)
            await session.flush()

            # Execute on Binance if keys are configured
            if settings.binance_order_execution_enabled:
                from app.services.binance_executor import BinanceExecutor
                executor = BinanceExecutor()
                result = await executor.execute_signal(session, signal)
                if not result.success:
                    return False, result.error_message

            return True, None

        except Exception as exc:
            logger.opt(exception=True).error("[ClaudeAgent] Execution error")
            return False, str(exc)

    async def _close_open_positions(
        self, session: AsyncSession, symbol: str
    ) -> tuple[bool, str | None]:
        """Mark all active signals for symbol as closed."""
        try:
            result = await session.execute(
                select(Signal).where(Signal.symbol == symbol, Signal.status == "active")
            )
            signals = result.scalars().all()
            for s in signals:
                s.status = "closed"
            await session.flush()
            return True, None
        except Exception as exc:
            logger.opt(exception=True).error("[ClaudeAgent] Close error")
            return False, str(exc)

    async def _get_daily_pnl(self, session: AsyncSession) -> float:
        """Sum P&L from outcomes created today."""
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        result = await session.execute(
            select(func.coalesce(func.sum(Outcome.pnl_usdt), 0)).where(
                Outcome.created_at >= today
            )
        )
        return float(result.scalar_one())

    async def _notify(self, decision: ClaudeDecision) -> None:
        """Send Telegram alert for the decision."""
        action_emoji = {
            "open_long": "📈",
            "open_short": "📉",
            "close": "🔒",
            "hold": "⏸",
        }.get(decision.action, "🤖")

        env = "TESTNET" if self._settings.binance_testnet else "MAINNET"
        exec_status = "✅ Executed" if decision.executed else ("⚠️ Signal only" if not decision.execution_error else f"❌ {decision.execution_error}")

        lines = [
            f"{action_emoji} <b>Claude Agent — {decision.action.upper()} {decision.symbol}</b>",
            f"<b>Environment:</b> {env}",
            f"<b>Confidence:</b> {decision.confidence:.0f}%",
            f"<b>Reasoning:</b> {decision.reasoning}",
        ]
        if decision.entry_price:
            lines.append(f"<b>Entry:</b> {float(decision.entry_price):.2f}")
        if decision.stop_loss:
            lines.append(f"<b>SL:</b> {float(decision.stop_loss):.2f}")
        if decision.take_profit:
            lines.append(f"<b>TP:</b> {float(decision.take_profit):.2f}")
        if decision.position_size:
            lines.append(f"<b>Size:</b> {float(decision.position_size):.4f}")
        lines.append(f"<b>Status:</b> {exec_status}")
        if decision.daily_pnl_at_decision is not None:
            lines.append(f"<b>Daily P&L:</b> ${float(decision.daily_pnl_at_decision):+.2f}")

        await self._notifier._send_message("\n".join(lines))
