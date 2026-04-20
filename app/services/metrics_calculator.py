"""Backtest metrics calculator for simulated trade results.

Computes key performance metrics from a list of SimulatedTrades:
win rate, profit factor, Sharpe ratio, max drawdown, and expectancy.
"""

import math
from dataclasses import dataclass
from decimal import Decimal

from app.services.trade_simulator import SimulatedTrade, TradeOutcome


@dataclass
class BacktestMetrics:
    """Aggregated performance metrics from a backtest run."""

    win_rate: Decimal
    profit_factor: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    expectancy: Decimal
    total_trades: int
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")
    sqn: Decimal = Decimal("0")
    kelly_criterion: Decimal = Decimal("0")


class MetricsCalculator:
    """Computes backtest performance metrics from simulated trades."""

    TRADING_DAYS_PER_YEAR = 252

    def compute(self, trades: list[SimulatedTrade]) -> BacktestMetrics:
        """Compute all backtest metrics from a list of simulated trades."""
        if not trades:
            return BacktestMetrics(
                win_rate=Decimal("0"),
                profit_factor=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                max_drawdown=Decimal("0"),
                expectancy=Decimal("0"),
                total_trades=0,
            )

        pnl_values = [float(t.pnl_pips) for t in trades]
        total = len(trades)

        wins = sum(
            1
            for t in trades
            if t.outcome in (TradeOutcome.TP1_HIT, TradeOutcome.TP2_HIT)
        )
        win_rate = wins / total

        gross_profit = sum(p for p in pnl_values if p > 0)
        gross_loss = abs(sum(p for p in pnl_values if p < 0))

        if gross_loss == 0:
            profit_factor = 9999.9999 if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss
            profit_factor = min(profit_factor, 9999.9999)

        expectancy = sum(pnl_values) / total

        if total < 2:
            sharpe_ratio = 0.0
        else:
            mean_pnl = sum(pnl_values) / total
            variance = sum((p - mean_pnl) ** 2 for p in pnl_values) / (total - 1)
            std_pnl = math.sqrt(variance)
            if std_pnl == 0:
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = (mean_pnl / std_pnl) * math.sqrt(self.TRADING_DAYS_PER_YEAR)

        max_drawdown = self._compute_max_drawdown(pnl_values)

        mean_pnl = sum(pnl_values) / total
        sortino_ratio = self._compute_sortino(pnl_values, mean_pnl)
        calmar_ratio = self._compute_calmar(pnl_values, max_drawdown)
        sqn = self._compute_sqn(pnl_values, expectancy)
        kelly_criterion = self._compute_kelly(win_rate, gross_profit, gross_loss, wins, total)

        return BacktestMetrics(
            win_rate=Decimal(str(round(win_rate, 4))),
            profit_factor=Decimal(str(round(profit_factor, 4))),
            sharpe_ratio=Decimal(str(round(sharpe_ratio, 4))),
            max_drawdown=Decimal(str(round(max_drawdown, 4))),
            expectancy=Decimal(str(round(expectancy, 4))),
            total_trades=total,
            sortino_ratio=Decimal(str(round(sortino_ratio, 4))),
            calmar_ratio=Decimal(str(round(calmar_ratio, 4))),
            sqn=Decimal(str(round(sqn, 4))),
            kelly_criterion=Decimal(str(round(kelly_criterion, 4))),
        )

    @staticmethod
    def _compute_max_drawdown(pnl_values: list[float]) -> float:
        """Compute maximum drawdown from a sequence of PnL values."""
        if not pnl_values:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnl_values:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def _compute_sortino(self, pnl_values: list[float], mean_pnl: float) -> float:
        """Sortino Ratio: mean_return / downside_std * √252. Penalises only downside vol."""
        downside = [p for p in pnl_values if p < 0]
        if len(downside) < 2:
            return 0.0
        downside_variance = sum(p ** 2 for p in downside) / len(downside)
        downside_std = math.sqrt(downside_variance)
        if downside_std == 0:
            return 0.0
        return (mean_pnl / downside_std) * math.sqrt(self.TRADING_DAYS_PER_YEAR)

    def _compute_calmar(self, pnl_values: list[float], max_drawdown: float) -> float:
        """Calmar Ratio: annualised_return / max_drawdown."""
        if max_drawdown == 0:
            return 0.0
        annual_return = sum(pnl_values) * self.TRADING_DAYS_PER_YEAR / len(pnl_values)
        return annual_return / max_drawdown

    @staticmethod
    def _compute_sqn(pnl_values: list[float], expectancy: float) -> float:
        """SQN (System Quality Number): (expectancy / std_expectancy) * √trades.
        SQN > 2.0 = good system, > 3.0 = excellent.
        """
        n = len(pnl_values)
        if n < 2:
            return 0.0
        variance = sum((p - expectancy) ** 2 for p in pnl_values) / (n - 1)
        std_exp = math.sqrt(variance)
        if std_exp == 0:
            return 0.0
        return (expectancy / std_exp) * math.sqrt(n)

    @staticmethod
    def _compute_kelly(
        win_rate: float,
        gross_profit: float,
        gross_loss: float,
        wins: int,
        total: int,
    ) -> float:
        """Half-Kelly criterion: optimal position fraction. WR - (LR / RR), halved for safety."""
        losses = total - wins
        if wins == 0 or losses == 0 or gross_loss == 0:
            return 0.0
        avg_win = gross_profit / wins
        avg_loss = gross_loss / losses
        if avg_loss == 0:
            return 0.0
        risk_reward = avg_win / avg_loss
        loss_rate = 1.0 - win_rate
        kelly = win_rate - (loss_rate / risk_reward)
        return max(0.0, kelly * 0.5)  # half-Kelly to avoid ruin
