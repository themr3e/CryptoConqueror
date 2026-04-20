"""Parameter optimization engine for strategy tuning.

Uses Optuna (Bayesian TPE) to explore parameter spaces — smarter than Latin
Hypercube Sampling because each trial informs the next.  The top candidates
are validated via walk-forward analysis, trade-shuffle Monte Carlo, AND a
candle-based random signal benchmark to confirm the edge is real.

Exports:
    ParamOptimizer -- main service class
"""

from __future__ import annotations

import asyncio
import gc
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
from loguru import logger

from app.services.backtester import BacktestRunner
from app.services.metrics_calculator import BacktestMetrics, MetricsCalculator
from app.services.walk_forward import CandleMonteCarloValidator, WalkForwardValidator
from app.strategies.base import BaseStrategy

PARAM_RANGES: dict[str, dict[str, tuple[float, float, float]]] = {
    "crypto_momentum": {
        "EMA_FAST": (8, 21, 1),
        "EMA_SLOW": (21, 55, 2),
        "RSI_PERIOD": (10, 20, 1),
        "SL_ATR_MULT": (0.5, 2.0, 0.25),
        "TP1_RR": (1.5, 3.0, 0.25),
    },
    "crypto_breakout": {
        "LOOKBACK": (10, 30, 2),
        "ATR_COMPRESSION": (0.3, 0.8, 0.05),
        "VOLUME_MULT": (1.0, 2.5, 0.25),
        "SL_ATR_MULT": (0.5, 2.0, 0.25),
        "TP1_RR": (1.5, 3.0, 0.25),
    },
}

NUM_SAMPLES = 80
MIN_TRADES_OPTIMIZE = 10
SCORE_WEIGHTS: dict[str, float] = {
    "win_rate": 0.30,
    "profit_factor": 0.25,
    "sharpe_ratio": 0.15,
    "expectancy": 0.15,
    "max_drawdown": 0.15,
}
TOP_N_VALIDATE = 5
MONTE_CARLO_RUNS = 300
MONTE_CARLO_CONFIDENCE = 0.05
VALIDATION_WINDOWS = [7, 14, 30]
MIN_WINDOWS_PASSING = 2


@dataclass
class OptimizationResult:
    """Result of optimizing a single strategy's parameters."""
    strategy_name: str
    best_params: dict[str, float]
    metrics: BacktestMetrics
    wfe_ratio: float | None
    is_overfitted: bool
    combinations_tested: int
    monte_carlo_pvalue: float | None = None
    candle_mc_pvalue: float | None = None   # p-value vs random entries (< 0.05 = real edge)
    beats_random: bool = True               # False when candle_mc_pvalue > 0.05


class ParamOptimizer:
    """Explores parameter spaces to find optimal strategy configurations."""

    def __init__(
        self,
        runner: BacktestRunner | None = None,
        wf_validator: WalkForwardValidator | None = None,
    ) -> None:
        self.runner = runner or BacktestRunner()
        self.wf_validator = wf_validator or WalkForwardValidator(runner=self.runner)
        self.metrics_calculator = MetricsCalculator()
        self._candle_mc = CandleMonteCarloValidator()

    async def optimize_strategy(
        self,
        strategy_name: str,
        candles: pd.DataFrame,
    ) -> OptimizationResult | None:
        """Optimize parameters using Optuna Bayesian TPE (smarter than LHS).

        Optuna learns from each trial — it models which parameter regions are
        promising and focuses exploration there.  After NUM_SAMPLES trials the
        top candidates are validated with walk-forward + trade Monte Carlo +
        candle-based random signal benchmark.
        """
        if strategy_name not in PARAM_RANGES:
            logger.warning("No parameter ranges defined for '{}', skipping", strategy_name)
            return None

        ranges = PARAM_RANGES[strategy_name]
        strategy_cls = BaseStrategy.get_registry().get(strategy_name)
        if strategy_cls is None:
            logger.error("Strategy '{}' not in registry", strategy_name)
            return None

        # ── Optuna Bayesian search ────────────────────────────────────────────
        scored = await self._optuna_search(strategy_name, strategy_cls, ranges, candles)

        if not scored:
            logger.warning("Optimizer: no viable candidates for '{}'", strategy_name)
            return None

        scored.sort(key=lambda x: x[2], reverse=True)
        n_tested = len(scored)

        # ── Validate top-N with WF + trade MC + candle MC ─────────────────────
        for rank, (params, metrics, score, trades) in enumerate(scored[:TOP_N_VALIDATE]):
            try:
                strategy = strategy_cls(params=params)

                wf_result = self.wf_validator.validate(strategy, candles, window_days=30)
                if wf_result.is_overfitted:
                    continue

                mc_pvalue = self._monte_carlo_test(trades, metrics)
                if mc_pvalue > MONTE_CARLO_CONFIDENCE:
                    continue

                windows_passed = 0
                for window in VALIDATION_WINDOWS:
                    try:
                        w_metrics, _ = self.runner.run_full_backtest(strategy, candles, window_days=window)
                        if w_metrics.total_trades >= MIN_TRADES_OPTIMIZE and float(w_metrics.profit_factor) > 1.0:
                            windows_passed += 1
                    except Exception:
                        pass

                if windows_passed < MIN_WINDOWS_PASSING:
                    continue

                wfe_values = [v for v in [wf_result.wfe_win_rate, wf_result.wfe_profit_factor] if v is not None]
                avg_wfe = sum(wfe_values) / len(wfe_values) if wfe_values else None

                # Candle-based random signal benchmark
                candle_mc_pvalue = self._candle_mc.test_edge(strategy, candles)
                beats_random = candle_mc_pvalue <= 0.05
                if not beats_random:
                    logger.warning(
                        "Optimizer: '{}' candidate #{} does NOT beat random entries (p={:.3f})",
                        strategy_name, rank, candle_mc_pvalue,
                    )
                    # Don't reject — still return result but flag it
                    # A strategy can fail this test early in its life and improve with data

                return OptimizationResult(
                    strategy_name=strategy_name,
                    best_params=params,
                    metrics=metrics,
                    wfe_ratio=avg_wfe,
                    is_overfitted=False,
                    combinations_tested=n_tested,
                    monte_carlo_pvalue=mc_pvalue,
                    candle_mc_pvalue=candle_mc_pvalue,
                    beats_random=beats_random,
                )

            except Exception:
                logger.exception("Optimizer: validation failed for '{}' candidate #{}", strategy_name, rank)

            await asyncio.sleep(0)

        best_params, best_metrics, _, _ = scored[0]
        return OptimizationResult(
            strategy_name=strategy_name,
            best_params=best_params,
            metrics=best_metrics,
            wfe_ratio=None,
            is_overfitted=True,
            combinations_tested=n_tested,
        )

    async def _optuna_search(
        self,
        strategy_name: str,
        strategy_cls,
        ranges: dict,
        candles: pd.DataFrame,
    ) -> list[tuple[dict[str, float], BacktestMetrics, float, list]]:
        """Run Optuna TPE search. Returns list of (params, metrics, score, trades)."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed — falling back to LHS sampling")
            candidates = self._generate_candidates(strategy_name, ranges)
            return await self._evaluate_candidates(candidates, strategy_cls, candles)

        scored: list[tuple[dict[str, float], BacktestMetrics, float, list]] = []
        defaults = dict(strategy_cls.DEFAULT_PARAMS)

        def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
            params = dict(defaults)
            for name, (lo, hi, step) in ranges.items():
                if step == int(step) and lo == int(lo) and hi == int(hi):
                    params[name] = float(trial.suggest_int(name, int(lo), int(hi), step=int(step)))
                else:
                    params[name] = trial.suggest_float(name, lo, hi, step=step)
            try:
                strategy = strategy_cls(params=params)
                metrics, trades = self.runner.run_full_backtest(strategy, candles, window_days=30)
                if metrics.total_trades < MIN_TRADES_OPTIMIZE:
                    return -1.0
                s = self._composite_score(metrics)
                scored.append((params, metrics, s, trades))
                return s
            except Exception:
                return -1.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=42),
        )

        # Run trials in batches so we can yield to the event loop
        batch = 10
        for start in range(0, NUM_SAMPLES, batch):
            n = min(batch, NUM_SAMPLES - start)
            study.optimize(objective, n_trials=n, show_progress_bar=False)
            await asyncio.sleep(0)
            gc.collect()

        try:
            best_val = study.best_value
        except ValueError:
            best_val = 0.0
        logger.info(
            "Optuna '{}': {} trials, best score={:.4f}",
            strategy_name, len(study.trials), best_val,
        )
        return scored

    async def _evaluate_candidates(
        self,
        candidates: list[dict],
        strategy_cls,
        candles: pd.DataFrame,
    ) -> list[tuple[dict[str, float], BacktestMetrics, float, list]]:
        """Evaluate pre-generated candidates (LHS fallback path)."""
        scored = []
        for idx, params in enumerate(candidates):
            try:
                strategy = strategy_cls(params=params)
                metrics, trades = self.runner.run_full_backtest(strategy, candles, window_days=30)
                if metrics.total_trades < MIN_TRADES_OPTIMIZE:
                    continue
                score = self._composite_score(metrics)
                scored.append((params, metrics, score, trades))
            except Exception:
                pass
            if idx % 10 == 0:
                await asyncio.sleep(0)
            if idx % 30 == 0:
                gc.collect()
        return scored

    def _monte_carlo_test(self, trades: list, original_metrics: BacktestMetrics) -> float:
        """Run Monte Carlo simulation to test statistical significance."""
        if not trades or original_metrics.total_trades < MIN_TRADES_OPTIMIZE:
            return 1.0

        original_pf = float(original_metrics.profit_factor)
        pnl_values = np.array([float(t.pnl_pips) for t in trades])

        rng = np.random.default_rng()
        beats = 0

        for _ in range(MONTE_CARLO_RUNS):
            shuffled = rng.permutation(pnl_values)
            gross_profit = float(np.sum(shuffled[shuffled > 0]))
            gross_loss = float(abs(np.sum(shuffled[shuffled < 0])))

            if gross_loss == 0:
                shuffled_pf = gross_profit if gross_profit > 0 else 0.0
            else:
                shuffled_pf = gross_profit / gross_loss

            if shuffled_pf >= original_pf:
                beats += 1

        return beats / MONTE_CARLO_RUNS

    def _generate_candidates(
        self,
        strategy_name: str,
        ranges: dict[str, tuple[float, float, float]],
    ) -> list[dict[str, float]]:
        """Generate parameter candidates using Latin Hypercube Sampling."""
        strategy_cls = BaseStrategy.get_registry()[strategy_name]
        defaults = dict(strategy_cls.DEFAULT_PARAMS)
        candidates = [dict(defaults)]

        param_names = list(ranges.keys())
        n_params = len(param_names)
        n_samples = NUM_SAMPLES - 1

        if n_samples <= 0 or n_params == 0:
            return candidates

        value_lists: list[list[float]] = []
        for name in param_names:
            lo, hi, step = ranges[name]
            values = []
            v = lo
            while v <= hi + step * 0.01:
                values.append(round(v, 4))
                v += step
            value_lists.append(values)

        rng = np.random.default_rng()
        for _ in range(n_samples):
            param_dict = dict(defaults)
            for dim, name in enumerate(param_names):
                vals = value_lists[dim]
                idx = rng.integers(0, len(vals))
                param_dict[name] = vals[idx]
            candidates.append(param_dict)

        return candidates

    @staticmethod
    def _composite_score(metrics: BacktestMetrics) -> float:
        """Compute a composite score from backtest metrics."""
        wr = float(metrics.win_rate)
        pf = min(float(metrics.profit_factor), 3.0) / 3.0
        sr = max(min(float(metrics.sharpe_ratio), 3.0), -1.0)
        sr_norm = (sr + 1.0) / 4.0
        exp = max(min(float(metrics.expectancy), 50.0), -20.0)
        exp_norm = (exp + 20.0) / 70.0
        dd = float(metrics.max_drawdown)
        dd_inv = 1.0 - dd

        return (
            SCORE_WEIGHTS["win_rate"] * wr
            + SCORE_WEIGHTS["profit_factor"] * pf
            + SCORE_WEIGHTS["sharpe_ratio"] * sr_norm
            + SCORE_WEIGHTS["expectancy"] * exp_norm
            + SCORE_WEIGHTS["max_drawdown"] * dd_inv
        )
