"""BacktestRunner: orchestrates strategy backtesting on rolling windows.

Runs any strategy's analyze() method on sliding windows of H1 candle data,
collects SimulatedTrades, and computes BacktestMetrics. Uses the EXACT same
strategy.analyze() code path as live signal generation -- no separate
backtest-only implementations.
"""

import pandas as pd
from loguru import logger

from app.services.metrics_calculator import BacktestMetrics, MetricsCalculator
from app.services.spread_model import SessionSpreadModel
from app.services.trade_simulator import SimulatedTrade, TradeSimulator
from app.strategies.base import BaseStrategy, InsufficientDataError


class BacktestRunner:
    """Runs strategies on rolling windows and collects simulated trades."""

    def __init__(
        self,
        simulator: TradeSimulator | None = None,
        spread_model: SessionSpreadModel | None = None,
        metrics_calculator: MetricsCalculator | None = None,
    ) -> None:
        self.simulator = simulator or TradeSimulator()
        self.spread_model = spread_model or SessionSpreadModel()
        self.metrics_calculator = metrics_calculator or MetricsCalculator()

    def run_rolling_backtest(
        self,
        strategy: BaseStrategy,
        candles: pd.DataFrame,
        window_days: int,
        step_days: int = 1,
    ) -> list[SimulatedTrade]:
        """Run a strategy on rolling windows and collect simulated trades."""
        window_candles = window_days * 24  # H1 = 24 candles/day
        step_candles = step_days * 24

        min_required = window_candles + TradeSimulator.MAX_BARS_FORWARD
        if len(candles) < min_required:
            logger.warning(
                "Insufficient candles for rolling backtest: "
                f"have {len(candles)}, need {min_required} "
                f"(window={window_days}d + {TradeSimulator.MAX_BARS_FORWARD} bars forward)"
            )
            return []

        trades: list[SimulatedTrade] = []

        for start_idx in range(
            0,
            len(candles) - window_candles - TradeSimulator.MAX_BARS_FORWARD,
            step_candles,
        ):
            end_idx = start_idx + window_candles
            window = candles.iloc[start_idx:end_idx].reset_index(drop=True)

            try:
                signals = strategy.generate_signals(window)
            except InsufficientDataError:
                logger.debug(
                    f"Skipping window at idx {start_idx}: insufficient data "
                    f"for strategy '{strategy.NAME}'"
                )
                continue
            except Exception:
                logger.exception(
                    f"Error in strategy '{strategy.NAME}' at window idx {start_idx}"
                )
                continue

            for signal in signals:
                try:
                    spread = self.spread_model.get_spread(signal.timestamp)
                    trade = self.simulator.simulate_trade(
                        signal, candles, end_idx - 1, spread
                    )
                    trades.append(trade)
                except Exception:
                    logger.exception(
                        f"Error simulating trade for signal at {signal.timestamp}"
                    )

        return trades

    def run_full_backtest(
        self,
        strategy: BaseStrategy,
        candles: pd.DataFrame,
        window_days: int,
        step_days: int = 1,
    ) -> tuple[BacktestMetrics, list[SimulatedTrade]]:
        """Run a rolling backtest and compute aggregate metrics."""
        trades = self.run_rolling_backtest(strategy, candles, window_days, step_days)
        metrics = self.metrics_calculator.compute(trades)

        logger.info(
            f"Backtest complete: strategy={strategy.NAME}, "
            f"window={window_days}d, "
            f"total_trades={metrics.total_trades}, "
            f"win_rate={metrics.win_rate}, "
            f"profit_factor={metrics.profit_factor}"
        )

        return metrics, trades

    def run_all_strategies(
        self,
        candles: pd.DataFrame,
        window_days_list: list[int] | None = None,
    ) -> dict[str, dict[int, tuple[BacktestMetrics, list[SimulatedTrade]]]]:
        """Run all registered strategies on multiple window sizes."""
        if window_days_list is None:
            window_days_list = [30, 60]

        registry = BaseStrategy.get_registry()
        results: dict[str, dict[int, tuple[BacktestMetrics, list[SimulatedTrade]]]] = {}

        for name, strategy_cls in registry.items():
            strategy = strategy_cls()
            results[name] = {}

            for window_days in window_days_list:
                try:
                    metrics, trades = self.run_full_backtest(strategy, candles, window_days)
                    results[name][window_days] = (metrics, trades)
                except Exception:
                    logger.exception(
                        f"Error running backtest for strategy '{name}' "
                        f"with window={window_days}d"
                    )

        return results
