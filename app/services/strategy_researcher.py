"""Autonomous strategy research and validation loop.

Every week Claude:
  1. Reviews its own recent trade history (wins, losses, patterns)
  2. Proposes a new trading strategy as Python code
  3. Runs an honest walk-forward blind backtest on 2 years of Binance data
  4. Decides automatically:
       win_rate >= 75% → auto-integrate, notify operator
       win_rate 60–74% → send results to operator, ask permission
       win_rate  < 60% → discard quietly

Claude cannot cheat:
  - The backtest uses only data available at each bar (no look-ahead)
  - Walk-forward training window is locked before blind testing
  - Claude never sees blind test data during generation

Generated strategies live in app/strategies/auto/ and are loaded
dynamically on startup.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.claude_decision import ClaudeDecision
from app.models.outcome import Outcome
from app.models.signal import Signal
from app.services.walk_forward_backtester import WalkForwardBacktester, WalkForwardResult
from app.strategies.base import BaseStrategy

_AUTO_DIR = Path(__file__).parent.parent / "strategies" / "auto"
_AUTO_WIN_RATE = 0.75
_ASK_WIN_RATE  = 0.60

_STRATEGY_TEMPLATE = '''
You are writing a new crypto futures trading strategy for a Binance Futures bot.

The strategy must:
1. Inherit from BaseStrategy
2. Have a unique NAME string (snake_case, e.g. "my_strategy_v1")
3. Implement generate_signals(candles: pd.DataFrame) -> list[CandidateSignal]
4. Only use data visible at the LAST bar of `candles` — NO look-ahead bias
5. Define DEFAULT_PARAMS dict with all tunable parameters
6. Define PARAM_GRID dict for walk-forward optimization (3-5 values per param)

CRITICAL — NO LOOK-AHEAD BIAS RULES:
- Never use .shift(-N) for negative N (that looks into the future)
- Never use .iloc[-1 + positive_number] to peek ahead
- Never use scipy filtfilt() — it uses future data. Use lfilter() instead
- The strategy receives candles UP TO the current bar only
- Only look backwards, never forwards

Available imports (already in scope):
    import pandas as pd
    import numpy as np
    from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
    from app.strategies.helpers.indicators import compute_atr

Write ONLY the Python class code. No markdown, no explanation, just the code.
Start with the imports, then the class.

Context about recent market behavior and what has been working/failing:
{context}

Based on this, propose a NEW strategy that addresses the observed patterns.
Think like a quant — what edge does this strategy exploit? Why would it work
on UNSEEN data, not just historical data?
'''


class StrategyResearcher:
    """Weekly autonomous strategy R&D loop."""

    async def run(self, session: AsyncSession) -> str:
        """Full research cycle: analyze → generate → backtest → decide.

        Returns a summary string for Telegram notification.
        """
        settings = get_settings()
        if not settings.anthropic_api_key:
            return "⚠️ ANTHROPIC_API_KEY not set — strategy research skipped."

        logger.info("[StrategyResearcher] Starting weekly research cycle")

        # Step 1: Build context from recent history
        context = await self._build_context(session)

        # Step 2: Ask Claude to write a new strategy
        code, strategy_name = await self._generate_strategy(context)
        if not code:
            return "⚠️ Strategy generation failed — Claude did not produce valid code."

        logger.info("[StrategyResearcher] Generated strategy: {}", strategy_name)

        # Step 3: Validate the code (syntax + structure)
        error = self._validate_code(code)
        if error:
            logger.warning("[StrategyResearcher] Code validation failed: {}", error)
            return f"⚠️ Generated strategy failed validation: {error}"

        # Step 4: Save to temp file and import
        strategy_cls = self._load_strategy(code, strategy_name)
        if strategy_cls is None:
            return "⚠️ Could not load generated strategy — import error."

        # Step 5: Walk-forward blind backtest
        logger.info("[StrategyResearcher] Running walk-forward backtest for {}", strategy_name)
        backtester = WalkForwardBacktester()
        try:
            result = await backtester.run(strategy_cls)
        except Exception:
            logger.opt(exception=True).error("[StrategyResearcher] Backtest failed")
            return f"⚠️ Backtest crashed for {strategy_name}."

        logger.info(
            "[StrategyResearcher] Backtest complete — win_rate={:.1f}% passed={}",
            result.win_rate * 100, result.passed,
        )

        # Step 6: Decision gate
        return await self._decide(result, code, strategy_name, session)

    # ── Context builder ───────────────────────────────────────────────────────

    async def _build_context(self, session: AsyncSession) -> str:
        """Build a summary of recent trade patterns for Claude to analyze."""
        from datetime import timedelta
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)

        # Recent outcomes
        outcomes_result = await session.execute(
            select(Outcome, Signal.symbol, Signal.direction,
                   Signal.entry_price, Signal.stop_loss, Signal.take_profit_1)
            .join(Signal, Outcome.signal_id == Signal.id)
            .where(Outcome.created_at >= week_ago)
            .order_by(Outcome.created_at.desc())
            .limit(50)
        )
        rows = outcomes_result.all()

        wins   = [r for r in rows if r[0].result in ("tp1_hit", "tp2_hit", "tp_hit")]
        losses = [r for r in rows if r[0].result in ("sl_hit",)]

        # Pattern analysis
        win_symbols  = [r[1] for r in wins]
        loss_symbols = [r[1] for r in losses]

        from collections import Counter
        top_win_symbols  = Counter(win_symbols).most_common(5)
        top_loss_symbols = Counter(loss_symbols).most_common(5)

        # Claude decisions that led to wins/losses
        decision_samples = []
        for outcome, symbol, direction, entry, sl, tp in rows[:10]:
            dec_result = await session.execute(
                select(ClaudeDecision)
                .where(
                    ClaudeDecision.symbol == symbol,
                    ClaudeDecision.action.in_(["open_long", "open_short"]),
                )
                .order_by(ClaudeDecision.created_at.desc())
                .limit(1)
            )
            decision = dec_result.scalar_one_or_none()
            reasoning = decision.reasoning[:100] if decision else "unknown"
            decision_samples.append(
                f"{'WIN' if outcome.result in ('tp1_hit','tp2_hit','tp_hit') else 'LOSS'} "
                f"{symbol} {direction} → {outcome.result} | {reasoning}"
            )

        context = f"""
Last 7 days: {len(rows)} closed trades | {len(wins)} wins | {len(losses)} losses
Win rate: {len(wins)/len(rows)*100:.0f}% (target: 75%+)

Top winning symbols: {top_win_symbols}
Top losing symbols:  {top_loss_symbols}

Sample decisions and outcomes:
{chr(10).join(decision_samples)}

Patterns to consider:
- Most losses happen when? (entering too early, wrong structure, against trend?)
- What conditions consistently produced wins?
- Is there a time-of-day pattern? (crypto trades 24/7 but has volume cycles)
- Are there specific price action patterns before wins?

Your job: design a strategy that avoids the loss patterns above and exploits the win patterns.
"""
        return context

    # ── Strategy generator ────────────────────────────────────────────────────

    async def _generate_strategy(
        self, context: str
    ) -> tuple[str, str]:
        """Ask Claude to write a new strategy. Returns (code, strategy_name)."""
        settings = get_settings()
        client   = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        prompt = _STRATEGY_TEMPLATE.format(context=context)

        try:
            message = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            code = message.content[0].text.strip()

            # Strip markdown code fences if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()

            # Extract strategy name from code
            name_match = re.search(r'NAME\s*=\s*["\']([^"\']+)["\']', code)
            strategy_name = name_match.group(1) if name_match else f"auto_{datetime.now().strftime('%Y%m%d_%H%M')}"

            return code, strategy_name

        except Exception:
            logger.opt(exception=True).error("[StrategyResearcher] Claude generation failed")
            return "", ""

    # ── Code validator ────────────────────────────────────────────────────────

    def _validate_code(self, code: str) -> str | None:
        """Return error string if code is invalid, None if OK."""
        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        # Must define a class inheriting BaseStrategy
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        if not classes:
            return "No class definition found"

        # Must have generate_signals method
        has_generate = any(
            isinstance(n, ast.FunctionDef) and n.name == "generate_signals"
            for n in ast.walk(tree)
        )
        if not has_generate:
            return "Missing generate_signals method"

        # Must have NAME attribute
        if "NAME" not in code:
            return "Missing NAME class attribute"

        # Look-ahead bias checks
        if re.search(r'\.shift\(-[1-9]', code):
            return "Look-ahead bias: .shift(-N) detected"
        if "filtfilt" in code:
            return "Look-ahead bias: filtfilt() uses future data, use lfilter() instead"

        # Block dangerous imports
        dangerous = ["subprocess", "os.system", "eval(", "exec(", "__import__"]
        for d in dangerous:
            if d in code:
                return f"Dangerous code pattern: {d}"

        return None

    # ── Dynamic loader ────────────────────────────────────────────────────────

    def _load_strategy(
        self, code: str, strategy_name: str
    ) -> type[BaseStrategy] | None:
        """Write code to a temp file and import it."""
        _AUTO_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-z0-9_]", "_", strategy_name.lower())
        file_path = _AUTO_DIR / f"_{safe_name}_test.py"

        # Add required imports if not present
        header = textwrap.dedent("""\
            from __future__ import annotations
            import pandas as pd
            import numpy as np
            from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
            from app.strategies.helpers.indicators import compute_atr
        """)
        full_code = header + "\n" + code

        try:
            file_path.write_text(full_code)
            spec   = importlib.util.spec_from_file_location(f"auto_{safe_name}", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the strategy class in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseStrategy)
                    and attr is not BaseStrategy
                    and hasattr(attr, "NAME")
                ):
                    return attr

            logger.warning("[StrategyResearcher] No BaseStrategy subclass found in generated code")
            return None

        except Exception:
            logger.opt(exception=True).error("[StrategyResearcher] Failed to load strategy")
            return None
        finally:
            # Clean up temp test file (persistent file written only if it passes)
            if file_path.exists():
                file_path.unlink()

    # ── Decision gate ─────────────────────────────────────────────────────────

    async def _decide(
        self,
        result: WalkForwardResult,
        code: str,
        strategy_name: str,
        session: AsyncSession,
    ) -> str:
        """Auto-integrate, ask, or discard based on win rate."""

        if result.win_rate >= _AUTO_WIN_RATE:
            # AUTO-INTEGRATE
            self._save_strategy(code, strategy_name)
            msg = (
                f"🚀 <b>New strategy auto-integrated!</b>\n\n"
                f"<b>Name:</b> {strategy_name}\n"
                f"{result.summary()}\n\n"
                f"Strategy is now LIVE — trading alongside existing strategies."
            )
            logger.info("[StrategyResearcher] AUTO-INTEGRATED: {}", strategy_name)
            return msg

        elif result.win_rate >= _ASK_WIN_RATE:
            # ASK OPERATOR — save code pending approval
            pending_path = _AUTO_DIR / f"_pending_{strategy_name}.py"
            pending_path.write_text(code)
            msg = (
                f"🔬 <b>New strategy needs your approval</b>\n\n"
                f"<b>Name:</b> {strategy_name}\n"
                f"{result.summary()}\n\n"
                f"Win rate {result.win_rate*100:.1f}% is above 60% but below the 75% auto-integrate threshold.\n"
                f"Reply <b>/approve_{strategy_name}</b> to integrate it, or ignore to discard."
            )
            logger.info("[StrategyResearcher] NEEDS APPROVAL: {} ({:.1f}%)", strategy_name, result.win_rate * 100)
            return msg

        else:
            # DISCARD
            msg = (
                f"🗑 <b>Strategy discarded (too weak)</b>\n\n"
                f"<b>Name:</b> {strategy_name}\n"
                f"{result.summary()}\n\n"
                f"Blind win rate {result.win_rate*100:.1f}% is below the 60% minimum. "
                f"Claude will try again next week with a different approach."
            )
            logger.info("[StrategyResearcher] DISCARDED: {} ({:.1f}%)", strategy_name, result.win_rate * 100)
            return msg

    def _save_strategy(self, code: str, strategy_name: str) -> None:
        """Permanently save an approved strategy to the auto directory."""
        _AUTO_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-z0-9_]", "_", strategy_name.lower())
        file_path = _AUTO_DIR / f"{safe_name}.py"

        header = textwrap.dedent("""\
            # Auto-generated strategy — passed walk-forward blind backtest
            # Do not edit manually — managed by StrategyResearcher
            from __future__ import annotations
            import pandas as pd
            import numpy as np
            from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
            from app.strategies.helpers.indicators import compute_atr
        """)
        file_path.write_text(header + "\n" + code)
        logger.info("[StrategyResearcher] Strategy saved to {}", file_path)
