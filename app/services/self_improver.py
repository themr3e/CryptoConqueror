"""Self-improvement loop for the Claude trading agent.

After every 10 closed trades, Claude reviews its own recent decisions,
identifies patterns in wins and losses, and stores actionable insights.
Those insights are injected into the next trading context so the bot
learns from its own history without human intervention.

Storage: PostgreSQL via ClaudeDecision table (no new migration needed).
Insights are stored as ClaudeDecision rows with action="self_analysis".
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import anthropic
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.claude_decision import ClaudeDecision
from app.models.outcome import Outcome
from app.models.signal import Signal


_ANALYSIS_PROMPT = """You are reviewing the last 10 closed trades of an autonomous crypto trading bot.
Your job: identify concrete, actionable patterns that explain WHY trades won or lost.

Respond with a JSON object only:
{
  "win_patterns": ["pattern 1", "pattern 2"],
  "loss_patterns": ["pattern 1", "pattern 2"],
  "rule_updates": ["specific rule or adjustment to apply going forward"],
  "symbols_to_avoid": ["symbol list based on recent losses"],
  "symbols_to_focus": ["symbols showing consistent wins"],
  "summary": "1-2 sentence summary of what the bot should do differently"
}

Be specific. "Avoid XYZUSDT when CVD is falling" is useful. "Trade better" is not.
"""

_LOSS_PROMPT = """A crypto futures trade just hit stop-loss. Write one actionable lesson.
Respond with JSON only — no markdown:
{"setup": "describe the entry setup", "context": "what market context was wrong", "result": "sl_hit", "lesson": "one-sentence rule to prevent this loss next time"}
Be specific. "Don't BUY BTCUSDT when H4 EMA is pointing down" beats "be more careful"."""


class SelfImprover:
    """Analyzes recent trade history and produces improvement insights."""

    TRIGGER_EVERY = 10      # Analyze after every 10th closed trade
    MAX_INSIGHT_AGE_DAYS = 7  # Only show insights from last 7 days

    async def maybe_analyze(self, session: AsyncSession) -> bool:
        """Run analysis if we've hit a multiple of TRIGGER_EVERY outcomes.

        Returns True if analysis was performed.
        """
        settings = get_settings()
        if not settings.anthropic_api_key:
            return False

        # Count total closed outcomes
        total_result = await session.execute(
            select(func.count(Outcome.id))
        )
        total = int(total_result.scalar_one() or 0)

        # Only trigger at multiples of 10
        if total == 0 or total % self.TRIGGER_EVERY != 0:
            return False

        # Don't re-analyze the same batch (check if we already analyzed at this count)
        existing = await session.execute(
            select(ClaudeDecision)
            .where(
                ClaudeDecision.action == "self_analysis",
                ClaudeDecision.reasoning.like(f"%total_outcomes={total}%"),
            )
            .limit(1)
        )
        if existing.scalar_one_or_none() is not None:
            return False

        logger.info("[SelfImprover] Triggering analysis at {} total outcomes", total)
        await self._run_analysis(session, total)
        return True

    async def get_latest_insight(self, session: AsyncSession) -> str:
        """Return the last 3 lessons (losses + analyses) injected into Claude's context."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.MAX_INSIGHT_AGE_DAYS)
        result = await session.execute(
            select(ClaudeDecision)
            .where(
                ClaudeDecision.action.in_(["self_analysis", "loss_lesson"]),
                ClaudeDecision.created_at >= cutoff,
            )
            .order_by(ClaudeDecision.created_at.desc())
            .limit(3)
        )
        rows = result.scalars().all()
        if not rows:
            return ""

        lines = ["🧠 [LAST 3 LESSONS — apply to this trade]"]
        for row in rows:
            try:
                if row.action == "loss_lesson":
                    data = json.loads(row.reasoning.split("lesson=", 1)[1])
                    lesson = data.get("lesson", "")
                    if lesson:
                        lines.append(f"  ⚠️ {lesson}")
                else:
                    data = json.loads(row.reasoning.split("json=", 1)[1])
                    if data.get("summary"):
                        lines.append(f"  📊 {data['summary']}")
                    for rule in data.get("rule_updates", [])[:2]:
                        lines.append(f"  ✏️  {rule}")
            except Exception:
                continue

        if len(lines) == 1:
            return ""

        lines.append("  Apply these rules in addition to the SLC framework.")
        return "\n".join(lines)

    async def analyze_loss(
        self,
        session: AsyncSession,
        signal: "Signal",
        outcome: "Outcome",
    ) -> None:
        """Write a micro-lesson after every stop-loss hit. Uses Haiku for low cost."""
        settings = get_settings()
        if not settings.anthropic_api_key:
            return

        trade_summary = (
            f"LOSS: {signal.symbol} {signal.direction} "
            f"entry={float(signal.entry_price):.4f} sl={float(signal.stop_loss):.4f} "
            f"pnl={float(outcome.pnl_usdt or 0):+.2f}"
        )
        prior_reasoning = await self._get_claude_reasoning(session, signal.id)
        if prior_reasoning:
            trade_summary += f" | Claude said: {prior_reasoning[:150]}"

        try:
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            message = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=_LOSS_PROMPT,
                messages=[{"role": "user", "content": trade_summary}],
            )
            raw = message.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)

            lesson_row = ClaudeDecision(
                symbol=signal.symbol,
                action="loss_lesson",
                reasoning=f"lesson={json.dumps(data)}",
                confidence=100.0,
                executed=False,
            )
            session.add(lesson_row)
            await session.commit()
            logger.info("[SelfImprover] Loss lesson — {}", data.get("lesson", "")[:100])
        except Exception:
            logger.opt(exception=True).warning("[SelfImprover] Loss lesson failed")

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _run_analysis(self, session: AsyncSession, total_outcomes: int) -> None:
        """Fetch last 10 outcomes + Claude reasoning, call Claude for analysis."""
        result = await session.execute(
            select(Outcome, Signal.symbol, Signal.direction,
                   Signal.entry_price, Signal.stop_loss, Signal.take_profit_1)
            .join(Signal, Outcome.signal_id == Signal.id)
            .order_by(Outcome.created_at.desc())
            .limit(10)
        )
        rows = result.all()
        if not rows:
            return

        # Build trade summary
        trade_lines = []
        for outcome, symbol, direction, entry, sl, tp in rows:
            pnl = float(outcome.pnl_usdt or 0)
            win = outcome.result in ("tp1_hit", "tp2_hit")
            # Find Claude's reasoning for this trade
            claude_reasoning = await self._get_claude_reasoning(session, outcome.signal_id)
            trade_lines.append(
                f"{'✅WIN' if win else '❌LOSS'} {symbol} {direction} "
                f"entry={float(entry):.4f} sl={float(sl):.4f} tp={float(tp):.4f} "
                f"pnl={pnl:+.2f} result={outcome.result}"
                + (f" | Claude said: {claude_reasoning[:120]}" if claude_reasoning else "")
            )

        trades_text = "\n".join(trade_lines)
        prompt = f"Last 10 trades:\n{trades_text}"

        try:
            client = anthropic.AsyncAnthropic(
                api_key=get_settings().anthropic_api_key
            )
            message = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=_ANALYSIS_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            # Validate it's parseable JSON
            data = json.loads(raw)

            # Store as a ClaudeDecision row with action="self_analysis"
            insight_row = ClaudeDecision(
                symbol="ALL",
                action="self_analysis",
                reasoning=f"total_outcomes={total_outcomes} json={json.dumps(data)}",
                confidence=100.0,
                executed=False,
            )
            session.add(insight_row)
            await session.commit()
            logger.info(
                "[SelfImprover] Analysis complete — summary: {}",
                data.get("summary", "")[:100],
            )

        except Exception:
            logger.opt(exception=True).error("[SelfImprover] Analysis failed")

    async def _get_claude_reasoning(
        self, session: AsyncSession, signal_id: int
    ) -> str:
        """Find the Claude decision that led to this signal being opened."""
        # ClaudeDecision doesn't link directly to Signal, match by symbol + time window
        signal_result = await session.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        signal = signal_result.scalar_one_or_none()
        if signal is None:
            return ""

        decision_result = await session.execute(
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
        decision = decision_result.scalar_one_or_none()
        return decision.reasoning if decision else ""
