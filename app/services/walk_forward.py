"""Walk-forward validation for overfitting detection."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd
from loguru import logger

from app.services.metrics_calculator import BacktestMetrics, MetricsCalculator
from app.strategies.base import BaseStrategy

DEGRADATION_THRESHOLD = 0.5  # OOS/IS ratio below 0.5 flags overfitting
MIN_OOS_TRADES = 5           # Minimum OOS trades required for detection
IS_RATIO = 0.8               # 80% in-sample, 20% out-of-sample


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
