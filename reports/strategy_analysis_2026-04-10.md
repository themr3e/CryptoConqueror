# Strategy Review — 2026-04-10 (window 15:35–16:41 UTC)

## 1. P&L summary (from logs; not a full ledger)

The pasted log window does **not** contain a trade-by-trade ledger, so exact P&L per
closed trade cannot be recovered. Only one hard outcome event is in the logs, plus
the per-symbol aggregates Claude Agent quoted from its context.

### Hard numbers from logs

| Source | Value |
|---|---|
| Only recorded outcome this window: `RENDERUSDT BUY sl_hit @ 2.003` | **-$0.0438** |
| Claude Agent account-state note (BTCUSDT reasoning, 15:50) | **"account already down $1975"** |
| Strategy `crypto_breakout` (id=14) 7-day rolling | win rate **21.03%** over **214** trades, profit factor **0** |
| Strategy `crypto_breakout` (id=14) 30-day rolling | win rate **15.26%** over **367** trades, profit factor **0** |

### Per-symbol P&L the agent cited in its reasoning

| Symbol | Wins/Trades | Agent-quoted P&L |
|---|---|---|
| ETHUSDT | 0/10 | **-$198.28** |
| AAVEUSDT | 1/10 | -$5.80 |
| LTCUSDT | 0/3 | -$1.25 |
| AVAXUSDT | 1/10 | -$0.61 |
| RENDERUSDT | 1/10 | -$0.19 |
| INJUSDT | 2/10 | -$0.09 |

**Sum of agent-quoted per-symbol losses** ≈ **-$206.18**. The rest of the
$1,975 drawdown must come from positions whose P&L was not quoted in the
agent prompt this window (FILUSDT, WLDUSDT, APTUSDT, and the 14 still-open longs
— see `reports/trades_2026-04-10.csv`).

> If you want the exact ledger, the authoritative source is the `crypto_trade_outcome`
> (or equivalent) table written by `app/services/crypto_outcome_detector.py`. Export
> it with:
> ```sql
> SELECT symbol, side, entry_price, exit_price, pnl_usdt, outcome, created_at
> FROM crypto_trade_outcome
> WHERE created_at >= now() - interval '30 days'
> ORDER BY created_at;
> ```

## 2. Trades worksheet

See `reports/trades_2026-04-10.csv` — 32 symbols covered, with columns:
`symbol, side, status, entry_price, win_rate, trades_sample, known_pnl_usdt,
agent_decision_1550, agent_decision_1620, agent_decision_1635, classification, notes`.

Highlights:
- **14 open long positions** (all BUY side): ADA, ARB, APT, ATOM, AVAX, DOT,
  INJ, JUP, LINK, NEAR, SOL, SUI, UNI, XRP — plus ETH, WLD, FIL implied open
  from context.
- **0 winning closed trades** observed in this window.
- **1 losing closed trade**: RENDERUSDT SL @ 2.003, -$0.0438.
- **Circuit breaker blocked 3 new entries**: DOTUSDT, UNIUSDT, RUNEUSDT
  (`signal_pipeline.py:125`).

## 3. What is wrong with the strategy concept

### Root cause: a long-only breakout strategy is running in the wrong regime

`strategy_selector.py:176` reports only one qualifying strategy, ranked:

```
crypto_breakout  score=0.4500  degraded=True  regime=LOW
```

That single line captures everything that's wrong.

### Diagnosis

1. **Breakout in LOW volatility regime is a concept mismatch.**
   Breakouts profit when volatility *expands* — price punches through a range
   and follow-through carries it. In LOW regime the opposite happens: price
   pokes above resistance, fails, and reverts. Every "breakout" becomes a
   false breakout into a chop-pay-the-spread trade. A 15–21% hit rate with
   **PF = 0** is the exact signature of this failure mode: you're paying
   fees + slippage, getting stopped repeatedly, and the occasional winner is
   too small to recoup.

2. **Long-only directional bias = catastrophic in a bear/range environment.**
   `signal_generator.py:232` keeps warning *"Directional bias detected: >75%
   of recent signals are BUY"*. There is no short side, so every downtick is
   an unhedged loss. Combined with item (1), you are systematically buying
   the top of micro-ranges.

3. **Degraded flag is live (`degraded=True`).** The selector already knows
   the strategy is underperforming and is running it anyway because it's
   the only candidate — `strategy_selector.py` has nothing to fall back on.
   When your only ranked strategy is flagged degraded, the correct action is
   **stop trading**, not "trade it degraded."

4. **Circuit breaker firing is a symptom, not a cure.** Three valid
   candidates (DOT/UNI/RUNE) were rejected because the breaker is hot. That
   is your risk system working — but the breaker will reset and the same
   losing strategy will resume entering. **The breaker is a speed bump, not
   a fix.**

5. **Stop-loss distance is almost certainly too tight.** With 367 trades at
   15.26% wr and PF=0, the expectancy math requires R:R of roughly **5.6:1
   just to break even** (breakeven R = (1-wr)/wr = 5.55). No breakout strategy
   entering on M15 pullbacks can realistically capture 5.6R. Either the
   take-profit is too close, the stop is too tight, or both.

6. **Entry filter is too loose.** `signal_generator.py:140` repeatedly
   returns 0 candidates from 350 candles — but the few it does return are
   almost all losers. That means the filter rejects 99% of bars and then
   picks the worst 1%: the filter is noise, not signal.

7. **Single-strategy portfolio = no diversification.** There is no
   mean-reversion, no trend-following, no pairs/stat-arb. When one strategy
   fails, the entire bot is exposed.

## 4. Proposed optimizations

### Immediate (today)

1. **Halt new live entries of `crypto_breakout` until 7-day wr > 40%.**
   Flip its `live_enabled` flag off in `strategy_selector.py` or the DB.
   Keep it in paper mode so it still generates the performance curve.

2. **Close the losing open longs that Claude Agent already flagged
   `close @ 60–75%` at 15:50.** The agent pivoted to `hold @ 0%` at 16:20,
   which just delays the inevitable. Re-run the close path for the 14
   symbols listed in the CSV.

3. **Widen all stops to ≥ 1.5 × ATR(14) on the entry timeframe** and require
   TP ≥ 2.5 × ATR (minimum 1.67 R:R). Current distances are implied too
   tight by the PF=0 result.

### Short term (this week)

4. **Add a second strategy for LOW regime.** Mean-reversion on RSI(2) +
   Bollinger Band touch is the textbook pair for breakout in LOW regime.
   `strategy_selector.py` already supports ranking multiple strategies —
   register it and let the selector pick whichever fits the regime.

5. **Enable the short side.** The >75% BUY bias warning is free alpha
   being left on the table. Mirror the breakout rules for short entries
   and require `signal_generator.validate()` to enforce a max 60% side
   imbalance.

6. **Add HTF trend filter.** Only take longs when D1 close > D1 EMA(50)
   *and* H4 close > H4 EMA(20). Only take shorts on the inverse. This
   alone typically lifts a 15% breakout wr to ~35–40%.

7. **Add volume confirmation.** Require entry bar volume > 1.5 × SMA(20)
   volume. Breakouts without volume are the false breakouts that are
   killing the PF.

### Medium term

8. **Walk-forward re-optimize the entry params** using
   `app/services/walk_forward.py` on the last 90 days, with PF and
   expectancy as joint objectives (not just wr).

9. **Per-symbol activation.** ETH (0/10), WLD (0/10), FIL (0/10), APT (0/8)
   should be auto-deactivated by `failure_tracker.py` after N consecutive
   losses on a symbol — not waiting for the global circuit breaker.

10. **Add a "degraded-strategy kill-switch" in `strategy_selector.py`:**
    if every qualifying strategy is `degraded=True`, return empty and let
    the pipeline go flat for that cycle. Running with no non-degraded
    strategies is how you got to -$1,975.

## 5. One-line summary

> You are running a long-only breakout strategy in a LOW-volatility,
> slightly-bearish regime, with stops that are too tight for the hit
> rate the strategy actually achieves. Every structural warning the
> system raised (degraded flag, >75% BUY bias, circuit breaker, PF=0)
> is telling you the same thing: **concept-regime mismatch**. Stop
> trading it live, add a LOW-regime mean-reversion counterpart, enable
> shorts, widen stops, and re-qualify before re-enabling.
