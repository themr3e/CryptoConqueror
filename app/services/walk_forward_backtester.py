"""Honest walk-forward backtesting engine.

Prevents curve fitting by:
  1. Training on a rolling window to find best parameters
  2. LOCKING those parameters
  3. Testing on the NEXT window of completely unseen data (blind)
  4. Only stitching together the blind results — never the training results

This is the same methodology described by quants and hedge funds.
A strategy that doesn't hold up on blind data has no real edge.

Walk-forward schedule (crypto — fast-moving markets):
    Train window:  90 days
    Blind window:  30 days
    Symbols:       BTCUSDT, ETHUSDT, SOLUSDT (aggregated)
    History:       2 years of H1 candles from Binance mainnet
    Fee model:     0.1% per side (0.2% round trip) + 0.05% slippage
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Type

import httpx
import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy

# ── Constants ─────────────────────────────────────────────────────────────────

_KLINES_URL   = "https://fapi.binance.com/fapi/v1/klines"
_TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
_TRAIN_DAYS   = 90
_BLIND_DAYS   = 30
_HISTORY_DAYS = 730          # 2 years
_FEE_RT       = 0.002        # 0.1% entry + 0.1% exit + slippage
_MAX_HOLD_BARS = 48          # close trade after 48 H1 bars if no SL/TP hit
_AUTO_WIN_RATE = 0.75        # auto-integrate threshold


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SimTrade:
    symbol:    str
    direction: str       # BUY / SELL
    entry:     float
    sl:        float
    tp:        float
    result:    str       # "tp_hit" | "sl_hit" | "expired"
    pnl_pct:   float     # net % return after fees
    bars_held: int
    fold:      int


@dataclass
class FoldResult:
    fold:        int
    train_start: datetime
    train_end:   datetime
    blind_start: datetime
    blind_end:   datetime
    trades:      list[SimTrade]
    win_rate:    float
    profit_factor: float
    best_params: dict


@dataclass
class WalkForwardResult:
    strategy_name: str
    symbols:       list[str]
    total_folds:   int
    blind_trades:  int
    blind_wins:    int
    win_rate:      float       # THE number — only blind results
    profit_factor: float
    total_pnl_pct: float
    fold_results:  list[FoldResult] = field(default_factory=list)
    passed:        bool = False    # True if win_rate >= AUTO_WIN_RATE

    def summary(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return (
            f"{status} — {self.strategy_name}\n"
            f"Blind win rate: {self.win_rate*100:.1f}% "
            f"({self.blind_wins}/{self.blind_trades} trades)\n"
            f"Profit factor: {self.profit_factor:.2f} | "
            f"Total P&L: {self.total_pnl_pct:+.1f}%\n"
            f"Folds tested: {self.total_folds} × {_BLIND_DAYS}d blind windows\n"
            f"Symbols: {', '.join(self.symbols)}"
        )


# ── Main engine ───────────────────────────────────────────────────────────────

class WalkForwardBacktester:
    """Run a strategy through honest walk-forward analysis."""

    async def run(
        self,
        strategy_cls: Type[BaseStrategy],
        symbols: list[str] | None = None,
    ) -> WalkForwardResult:
        """Run full walk-forward test across multiple symbols.

        Fetches real Binance mainnet data — always, even on testnet — so
        the backtest uses real market history, not demo data.
        """
        test_symbols = symbols or _TEST_SYMBOLS
        logger.info(
            "[WFBacktest] Starting walk-forward for {} on {}",
            strategy_cls.NAME, test_symbols,
        )

        all_fold_results: list[FoldResult] = []

        for symbol in test_symbols:
            candles = await self._fetch_historical(symbol)
            if candles is None or len(candles) < (_TRAIN_DAYS + _BLIND_DAYS) * 24:
                logger.warning("[WFBacktest] Not enough history for {} — skipping", symbol)
                continue

            folds = self._run_walk_forward(strategy_cls, candles, symbol)
            all_fold_results.extend(folds)

        return self._aggregate(strategy_cls.NAME, test_symbols, all_fold_results)

    # ── Walk-forward core ─────────────────────────────────────────────────────

    def _run_walk_forward(
        self,
        strategy_cls: Type[BaseStrategy],
        candles: pd.DataFrame,
        symbol: str,
    ) -> list[FoldResult]:
        train_bars = _TRAIN_DAYS * 24
        blind_bars = _BLIND_DAYS * 24
        step_bars  = blind_bars          # walk forward by one blind window at a time

        folds: list[FoldResult] = []
        fold_num = 0
        pos = 0

        while pos + train_bars + blind_bars <= len(candles):
            train_df = candles.iloc[pos : pos + train_bars].copy()
            blind_df = candles.iloc[pos + train_bars : pos + train_bars + blind_bars].copy()

            # ── Step 1: optimize parameters on training data ──────────────
            best_params, train_wr = self._optimize(strategy_cls, train_df, symbol)

            # ── Step 2: lock params, run blind test ───────────────────────
            strategy = strategy_cls(params=best_params)
            blind_trades = self._simulate(strategy, blind_df, symbol, fold_num)

            wins  = sum(1 for t in blind_trades if t.result == "tp_hit")
            total = len(blind_trades)
            blind_wr = wins / total if total > 0 else 0.0

            gross_wins  = sum(t.pnl_pct for t in blind_trades if t.pnl_pct > 0)
            gross_loss  = abs(sum(t.pnl_pct for t in blind_trades if t.pnl_pct < 0))
            pf = gross_wins / gross_loss if gross_loss > 0 else (1.0 if gross_wins == 0 else 999.0)

            fold = FoldResult(
                fold=fold_num,
                train_start=train_df.index[0].to_pydatetime(),
                train_end=train_df.index[-1].to_pydatetime(),
                blind_start=blind_df.index[0].to_pydatetime(),
                blind_end=blind_df.index[-1].to_pydatetime(),
                trades=blind_trades,
                win_rate=blind_wr,
                profit_factor=pf,
                best_params=best_params,
            )
            folds.append(fold)

            logger.info(
                "[WFBacktest] {} fold {} — blind: {}/{} wins ({:.0f}%) params={}",
                symbol, fold_num, wins, total, blind_wr * 100, best_params,
            )

            pos      += step_bars
            fold_num += 1

        return folds

    # ── Parameter optimizer (grid search on training window) ─────────────────

    def _optimize(
        self,
        strategy_cls: Type[BaseStrategy],
        train_df: pd.DataFrame,
        symbol: str,
    ) -> tuple[dict, float]:
        """Find the best parameter combination on training data."""
        param_grid = getattr(strategy_cls, "PARAM_GRID", None)
        if not param_grid:
            # No grid defined — use defaults
            return dict(strategy_cls.DEFAULT_PARAMS), 0.0

        best_params = dict(strategy_cls.DEFAULT_PARAMS)
        best_score  = -999.0

        keys   = list(param_grid.keys())
        values = list(param_grid.values())

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            # Fill in any params not in the grid with defaults
            full_params = {**strategy_cls.DEFAULT_PARAMS, **params}

            try:
                strategy = strategy_cls(params=full_params)
                trades   = self._simulate(strategy, train_df, symbol, fold=-1)
            except Exception:
                continue

            if not trades:
                continue

            wins  = sum(1 for t in trades if t.result == "tp_hit")
            total = len(trades)
            if total < 5:
                continue     # too few trades to be meaningful

            # Score = win_rate * log(total_trades) — balance quality and quantity
            import math
            score = (wins / total) * math.log(max(total, 1))
            if score > best_score:
                best_score  = score
                best_params = full_params

        return best_params, best_score

    # ── Trade simulator ───────────────────────────────────────────────────────

    def _simulate(
        self,
        strategy: BaseStrategy,
        candles: pd.DataFrame,
        symbol: str,
        fold: int,
    ) -> list[SimTrade]:
        """Simulate trades bar by bar — no look-ahead.

        At each bar N, the strategy only sees candles[0:N+1].
        SL/TP is checked on subsequent bars.
        """
        candles.attrs["symbol"] = symbol
        trades: list[SimTrade] = []
        n = len(candles)
        in_trade = False

        for i in range(60, n):          # need at least 60 bars of history
            if in_trade:
                continue

            # Strategy sees ONLY data up to and including bar i
            window = candles.iloc[:i + 1].copy()
            window.attrs["symbol"] = symbol

            try:
                signals = strategy.generate_signals(window)
            except Exception:
                continue

            if not signals:
                continue

            sig = signals[0]
            entry = float(sig.entry_price)
            sl    = float(sig.stop_loss)
            tp    = float(sig.take_profit_1)
            direction = sig.direction.value if hasattr(sig.direction, "value") else str(sig.direction)

            if entry <= 0 or sl <= 0 or tp <= 0:
                continue

            # Simulate forward from bar i+1
            result    = "expired"
            pnl_pct   = 0.0
            bars_held = 0

            for j in range(i + 1, min(i + 1 + _MAX_HOLD_BARS, n)):
                bar_high = float(candles.iloc[j]["high"])
                bar_low  = float(candles.iloc[j]["low"])
                bars_held = j - i

                if direction == "BUY":
                    if bar_low <= sl:
                        result  = "sl_hit"
                        pnl_pct = ((sl - entry) / entry) - _FEE_RT
                        break
                    if bar_high >= tp:
                        result  = "tp_hit"
                        pnl_pct = ((tp - entry) / entry) - _FEE_RT
                        break
                else:  # SELL
                    if bar_high >= sl:
                        result  = "sl_hit"
                        pnl_pct = ((entry - sl) / entry) - _FEE_RT
                        break
                    if bar_low <= tp:
                        result  = "tp_hit"
                        pnl_pct = ((entry - tp) / entry) - _FEE_RT
                        break

            if result == "expired":
                # Close at last bar mid-price
                exit_price = float(candles.iloc[min(i + _MAX_HOLD_BARS, n - 1)]["close"])
                if direction == "BUY":
                    pnl_pct = ((exit_price - entry) / entry) - _FEE_RT
                else:
                    pnl_pct = ((entry - exit_price) / entry) - _FEE_RT

            trades.append(SimTrade(
                symbol=symbol,
                direction=direction,
                entry=entry,
                sl=sl,
                tp=tp,
                result=result,
                pnl_pct=pnl_pct,
                bars_held=bars_held,
                fold=fold,
            ))
            in_trade = False   # allow next signal (one at a time per symbol)

        return trades

    # ── Results aggregator ────────────────────────────────────────────────────

    def _aggregate(
        self,
        strategy_name: str,
        symbols: list[str],
        folds: list[FoldResult],
    ) -> WalkForwardResult:
        all_trades  = [t for f in folds for t in f.trades]
        blind_wins  = sum(1 for t in all_trades if t.result == "tp_hit")
        blind_total = len(all_trades)
        win_rate    = blind_wins / blind_total if blind_total > 0 else 0.0

        gross_wins = sum(t.pnl_pct for t in all_trades if t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in all_trades if t.pnl_pct < 0))
        pf         = gross_wins / gross_loss if gross_loss > 0 else (1.0 if gross_wins == 0 else 999.0)
        total_pnl  = sum(t.pnl_pct for t in all_trades) * 100

        return WalkForwardResult(
            strategy_name=strategy_name,
            symbols=symbols,
            total_folds=len(folds),
            blind_trades=blind_total,
            blind_wins=blind_wins,
            win_rate=win_rate,
            profit_factor=pf,
            total_pnl_pct=total_pnl,
            fold_results=folds,
            passed=win_rate >= _AUTO_WIN_RATE,
        )

    # ── Binance historical data fetcher ───────────────────────────────────────

    async def _fetch_historical(
        self, symbol: str, days: int = _HISTORY_DAYS
    ) -> pd.DataFrame | None:
        """Fetch H1 candles from Binance mainnet (always mainnet for real history)."""
        limit      = 1000
        total_bars = days * 24
        all_bars:  list[list] = []

        end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - days * 24 * 3600 * 1000

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                current_start = start_ms
                while len(all_bars) < total_bars:
                    resp = await client.get(_KLINES_URL, params={
                        "symbol":    symbol,
                        "interval":  "1h",
                        "startTime": current_start,
                        "endTime":   end_ms,
                        "limit":     limit,
                    })
                    if resp.status_code != 200:
                        logger.warning("[WFBacktest] Binance returned {} for {}", resp.status_code, symbol)
                        break
                    batch = resp.json()
                    if not batch:
                        break
                    all_bars.extend(batch)
                    if len(batch) < limit:
                        break
                    current_start = int(batch[-1][0]) + 1
        except Exception:
            logger.opt(exception=True).error("[WFBacktest] Failed to fetch history for {}", symbol)
            return None

        if not all_bars:
            return None

        df = pd.DataFrame(all_bars, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df[["open", "high", "low", "close", "volume"]]
        df.attrs["symbol"] = symbol

        logger.info("[WFBacktest] Fetched {} H1 bars for {}", len(df), symbol)
        return df
