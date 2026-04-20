"""Walk-forward validation and candle Monte Carlo for overfitting detection."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
from loguru import logger

from app.services.metrics_calculator import BacktestMetrics, MetricsCalculator
from app.strategies.base import BaseStrategy

DEGRADATION_THRESHOLD = 0.5  # OOS/IS ratio below 0.5 flags overfitting
MIN_OOS_TRADES = 5           # Minimum OOS trades required for detection
IS_RATIO = 0.8               # 80% in-sample, 20% out-of-sample

CANDLE_MC_RUNS = 500         # random strategy runs for candle-based Monte Carlo
CANDLE_MC_SIGNIFICANCE = 0.05  # p-value threshold — must beat 95th pct of random


@dataclass
class WalkForwardResult:
    """Results from a walk-forward validation run."""
    is_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics
    is_overfitted: bool
    wfe_win_rate: float | None
    wfe_profit_factor: float | None


class WalkForwardValidator:
    """Validates strategies through 80/20 in-sample / out-of-sample split."""

    def __init__(self, runner=None) -> None:
        from app.services.backtester import BacktestRunner
        self.runner = runner or BacktestRunner()
        self.metrics_calculator = MetricsCalculator()

    def validate(
        self,
        strategy: BaseStrategy,
        candles: pd.DataFrame,
        window_days: int = 30,
    ) -> WalkForwardResult:
        """Run walk-forward validation on candle data.

        Args:
            strategy: Strategy instance to validate.
            candles: Full H1 OHLC DataFrame (chronological order).
            window_days: Window size for each backtest.

        Returns:
            WalkForwardResult with IS/OOS metrics and overfitting flags.
        """
        split_idx = int(len(candles) * IS_RATIO)
        is_candles = candles.iloc[:split_idx].reset_index(drop=True)
        oos_candles = candles.iloc[split_idx:].reset_index(drop=True)

        logger.info(
            "Walk-forward split: IS={} bars, OOS={} bars for '{}'",
            len(is_candles),
            len(oos_candles),
            strategy.name,
        )

        is_metrics, _ = self.runner.run_full_backtest(strategy, is_candles, window_days)
        oos_metrics, _ = self.runner.run_full_backtest(strategy, oos_candles, window_days)

        if oos_metrics.total_trades < MIN_OOS_TRADES:
            logger.info(
                "Walk-forward: insufficient OOS trades ({}) for '{}', skipping overfitting check",
                oos_metrics.total_trades,
                strategy.name,
            )
            return WalkForwardResult(
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
                is_overfitted=False,
                wfe_win_rate=None,
                wfe_profit_factor=None,
            )

        is_wr = float(is_metrics.win_rate)
        oos_wr = float(oos_metrics.win_rate)
        is_pf = float(is_metrics.profit_factor)
        oos_pf = float(oos_metrics.profit_factor)

        wfe_wr = (oos_wr / is_wr) if is_wr > 0 else None
        wfe_pf = (oos_pf / is_pf) if is_pf > 0 else None

        is_overfitted = (
            (wfe_wr is not None and wfe_wr < DEGRADATION_THRESHOLD) or
            (wfe_pf is not None and wfe_pf < DEGRADATION_THRESHOLD)
        )

        logger.info(
            "Walk-forward '{}': IS wr={:.3f} pf={:.3f} | OOS wr={:.3f} pf={:.3f} | "
            "wfe_wr={} wfe_pf={} | overfitted={}",
            strategy.name,
            is_wr, is_pf,
            oos_wr, oos_pf,
            f"{wfe_wr:.3f}" if wfe_wr is not None else "N/A",
            f"{wfe_pf:.3f}" if wfe_pf is not None else "N/A",
            is_overfitted,
        )

        return WalkForwardResult(
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            is_overfitted=is_overfitted,
            wfe_win_rate=wfe_wr,
            wfe_profit_factor=wfe_pf,
        )


class CandleMonteCarloValidator:
    """Test whether a strategy's edge is real by comparing it against random entries.

    Generates CANDLE_MC_RUNS random strategies that enter at random bars with the
    same SL/TP structure as our strategy.  If our strategy cannot beat the 95th
    percentile of these random strategies, it has no real entry edge.

    This is stronger than trade-shuffle Monte Carlo, which only tests order
    dependency — this tests whether the ENTRY LOGIC itself has any value.
    """

    def test_edge(
        self,
        strategy: BaseStrategy,
        candles: pd.DataFrame,
        n_runs: int = CANDLE_MC_RUNS,
    ) -> float:
        """Return p-value: fraction of random strategies whose PF ≥ ours.

        p-value < 0.05 → strategy beats 95% of random → edge is real.
        p-value > 0.05 → cannot distinguish from random → no real edge.
        """
        from app.services.backtester import BacktestRunner
        runner = BacktestRunner()

        # Run our strategy to get its profit factor
        try:
            our_metrics, our_trades = runner.run_full_backtest(strategy, candles, window_days=30)
        except Exception:
            return 1.0

        if not our_trades or our_metrics.total_trades < 5:
            return 1.0

        our_pf = float(our_metrics.profit_factor)

        # Build numpy arrays for fast random simulation
        closes = candles["close"].astype(float).values
        highs  = candles["high"].astype(float).values
        lows   = candles["low"].astype(float).values
        n_bars = len(closes)

        # Estimate ATR (14-period) for realistic SL sizing
        atr_arr = self._compute_atr_np(highs, lows, closes, period=14)

        rr = float(strategy.params.get("TP1_RR", strategy.params.get("TP_RR", 2.0)))
        sl_mult = float(strategy.params.get("ATR_SL_MULT", strategy.params.get("SL_ATR_MULT", 1.0)))
        n_trades = our_metrics.total_trades

        rng = np.random.default_rng(seed=42)
        random_pfs: list[float] = []

        for _ in range(n_runs):
            gross_profit = 0.0
            gross_loss   = 0.0

            # Random entry bars (avoid first/last 30 bars to have room for simulation)
            entry_bars = rng.integers(30, max(31, n_bars - 20), size=n_trades)
            directions = rng.choice(["BUY", "SELL"], size=n_trades)

            for bar, direction in zip(entry_bars, directions):
                atr = atr_arr[bar]
                if np.isnan(atr) or atr <= 0:
                    continue

                entry = closes[bar]
                sl_dist = atr * sl_mult
                tp_dist = sl_dist * rr

                # Simulate forward — first of SL/TP hit wins
                for j in range(1, 21):
                    if bar + j >= n_bars:
                        break
                    h = highs[bar + j]
                    lo = lows[bar + j]
                    if direction == "BUY":
                        if lo <= entry - sl_dist:
                            gross_loss += sl_dist
                            break
                        if h >= entry + tp_dist:
                            gross_profit += tp_dist
                            break
                    else:
                        if h >= entry + sl_dist:
                            gross_loss += sl_dist
                            break
                        if lo <= entry - tp_dist:
                            gross_profit += tp_dist
                            break

            rand_pf = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0.0)
            random_pfs.append(rand_pf)

        if not random_pfs:
            return 1.0

        beats = sum(1 for pf in random_pfs if pf >= our_pf)
        p_value = beats / len(random_pfs)

        logger.info(
            "CandleMC '{}': our_pf={:.3f}  random_p95={:.3f}  p_value={:.3f}  edge={}",
            strategy.name,
            our_pf,
            float(np.percentile(random_pfs, 95)),
            p_value,
            "REAL" if p_value <= CANDLE_MC_SIGNIFICANCE else "RANDOM",
        )
        return p_value

    @staticmethod
    def _compute_atr_np(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Wilder ATR using numpy — avoids pandas overhead inside the MC loop."""
        n = len(closes)
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.full(n, np.nan)
        if n >= period:
            atr[period - 1] = tr[:period].mean()
            alpha = 1.0 / period
            for i in range(period, n):
                atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha

        return atr
