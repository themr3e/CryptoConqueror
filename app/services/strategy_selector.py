"""Strategy selector service: ranking and volatility regime detection.

Ranks all registered strategies by composite score (win rate, profit factor,
Sharpe ratio, expectancy, max drawdown) and detects the current market
volatility regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from loguru import logger
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.backtest_result import BacktestResult
from app.models.candle import Candle
from app.models.strategy import Strategy
from app.models.strategy_performance import StrategyPerformance

MIN_TRADES_QUALIFY = 8
MIN_LIVE_SIGNALS_BLEND = 5

SCORE_WEIGHTS = {
    "win_rate": 0.30,
    "profit_factor": 0.25,
    "sharpe_ratio": 0.15,
    "expectancy": 0.15,
    "max_drawdown": 0.15,
}

REGIME_MODIFIERS = {
    "breakout_expansion": {"HIGH": -0.10},
    "trend_continuation": {"LOW": -0.10},
}


class VolatilityRegime(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class StrategyScore:
    """Result of composite scoring for a single strategy."""
    strategy_name: str
    composite_score: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    expectancy: float
    max_drawdown: float
    total_trades: int
    regime: VolatilityRegime
    is_degraded: bool


class StrategySelector:
    """Selects and ranks strategies based on backtest performance metrics."""

    async def select_best(
        self,
        session: AsyncSession,
        asset_class: str = "forex",
    ) -> StrategyScore | None:
        """Return the highest-scoring qualifying strategy, or None."""
        ranked = await self.select_all_ranked(session, asset_class=asset_class)
        return ranked[0] if ranked else None

    async def select_all_ranked(
        self,
        session: AsyncSession,
        asset_class: str = "forex",
    ) -> list[StrategyScore]:
        """Return all qualifying strategies ranked by composite score.

        Args:
            session:     Async DB session.
            asset_class: Filter strategies by asset class.
                         ``"forex"`` for XAUUSD, ``"crypto_futures"`` for BTC/ETH.
        """
        # Use a representative symbol for ATR/regime detection per asset class
        regime_symbol = "XAUUSD" if asset_class == "forex" else "BTCUSDT"
        regime = await self._detect_volatility_regime(session, symbol=regime_symbol)

        stmt = (
            select(Strategy.name, BacktestResult)
            .join(Strategy, BacktestResult.strategy_id == Strategy.id)
            .where(
                and_(
                    Strategy.is_active.is_(True),
                    Strategy.asset_class == asset_class,
                    BacktestResult.is_walk_forward.isnot(True),
                    BacktestResult.window_days == 30,
                )
            )
            .order_by(BacktestResult.created_at.desc())
        )
        result = await session.execute(stmt)
        rows = result.all()

        # Group by strategy name, keep latest result
        latest: dict[str, BacktestResult] = {}
        names: dict[str, str] = {}
        for name, bt in rows:
            if name not in latest:
                latest[name] = bt
                names[bt.strategy_id] = name

        if not latest:
            logger.warning("StrategySelector: no qualifying backtest results found")
            return []

        scores = []
        for strategy_name, bt in latest.items():
            if bt.total_trades < MIN_TRADES_QUALIFY:
                logger.debug(
                    "Strategy '{}' skipped: only {} trades (need {})",
                    strategy_name, bt.total_trades, MIN_TRADES_QUALIFY,
                )
                continue

            wr = float(bt.win_rate or 0)
            pf = float(bt.profit_factor or 0)
            sr = float(bt.sharpe_ratio or 0)
            exp = float(bt.expectancy or 0)
            dd = float(bt.max_drawdown or 0)

            # Check degradation
            is_degraded = pf < 1.0

            # Blend with live performance if available
            live_perf = await self._get_live_performance(session, bt.strategy_id)
            if live_perf and live_perf.total_signals >= MIN_LIVE_SIGNALS_BLEND:
                live_wr = float(live_perf.win_rate)
                live_pf = float(live_perf.profit_factor)
                wr = wr * 0.7 + live_wr * 0.3
                pf = pf * 0.7 + live_pf * 0.3

            # Normalize to [0, 1]
            all_wrs = [float(b.win_rate or 0) for _, b in latest.items()]
            all_pfs = [min(float(b.profit_factor or 0), 3.0) for _, b in latest.items()]

            wr_norm = self._normalize(wr, min(all_wrs), max(all_wrs))
            pf_norm = self._normalize(min(pf, 3.0), min(all_pfs), max(all_pfs))
            sr_norm = max(min((sr + 1.0) / 4.0, 1.0), 0.0)
            exp_norm = max(min((exp + 20.0) / 70.0, 1.0), 0.0)
            dd_inv = 1.0 - min(dd, 1.0)

            score = (
                SCORE_WEIGHTS["win_rate"] * wr_norm
                + SCORE_WEIGHTS["profit_factor"] * pf_norm
                + SCORE_WEIGHTS["sharpe_ratio"] * sr_norm
                + SCORE_WEIGHTS["expectancy"] * exp_norm
                + SCORE_WEIGHTS["max_drawdown"] * dd_inv
            )

            # Apply regime modifier
            modifier = REGIME_MODIFIERS.get(strategy_name, {}).get(regime.value, 0.0)
            score += modifier

            scores.append(StrategyScore(
                strategy_name=strategy_name,
                composite_score=max(0.0, score),
                win_rate=wr,
                profit_factor=pf,
                sharpe_ratio=sr,
                expectancy=exp,
                max_drawdown=dd,
                total_trades=bt.total_trades,
                regime=regime,
                is_degraded=is_degraded,
            ))

        scores.sort(key=lambda s: s.composite_score, reverse=True)

        logger.info(
            "StrategySelector: {} strategies ranked (regime={})",
            len(scores), regime.value,
        )

        return scores

    async def check_h4_confluence(
        self,
        session: AsyncSession,
        direction: str,
        symbol: str = "XAUUSD",
    ) -> bool:
        """Check if H4 EMA-50/200 confirms the signal direction for the given symbol."""
        try:
            stmt = (
                select(Candle)
                .where(
                    and_(
                        Candle.symbol == symbol,
                        Candle.timeframe == "H4",
                    )
                )
                .order_by(Candle.timestamp.desc())
                .limit(250)
            )
            result = await session.execute(stmt)
            candles = result.scalars().all()

            if len(candles) < 200:
                return False

            closes = [float(c.close) for c in reversed(candles)]

            ema50 = self._ema(closes, 50)
            ema200 = self._ema(closes, 200)

            if direction == "BUY":
                return ema50[-1] > ema200[-1]
            else:
                return ema50[-1] < ema200[-1]
        except Exception:
            return False

    async def _detect_volatility_regime(
        self,
        session: AsyncSession,
        symbol: str = "XAUUSD",
    ) -> VolatilityRegime:
        """Detect current volatility regime using ATR percentile for the given symbol."""
        try:
            from app.strategies.helpers.indicators import compute_atr
            import pandas as pd

            stmt = (
                select(Candle.high, Candle.low, Candle.close)
                .where(
                    and_(
                        Candle.symbol == symbol,
                        Candle.timeframe == "H1",
                    )
                )
                .order_by(Candle.timestamp.desc())
                .limit(720)
            )
            result = await session.execute(stmt)
            rows = list(reversed(result.all()))

            if len(rows) < 50:
                return VolatilityRegime.MEDIUM

            highs = pd.Series([float(r[0]) for r in rows])
            lows = pd.Series([float(r[1]) for r in rows])
            closes = pd.Series([float(r[2]) for r in rows])

            atr = compute_atr(highs, lows, closes, 14).dropna()
            if atr.empty:
                return VolatilityRegime.MEDIUM

            current = float(atr.iloc[-1])
            p33 = float(atr.quantile(0.33))
            p67 = float(atr.quantile(0.67))

            if current <= p33:
                return VolatilityRegime.LOW
            elif current >= p67:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.MEDIUM

        except Exception:
            return VolatilityRegime.MEDIUM

    async def _get_live_performance(self, session: AsyncSession, strategy_id: int):
        """Get the 30d live performance record for a strategy."""
        try:
            stmt = select(StrategyPerformance).where(
                and_(
                    StrategyPerformance.strategy_id == strategy_id,
                    StrategyPerformance.period == "30d",
                )
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception:
            return None

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1] range."""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    @staticmethod
    def _ema(closes: list[float], period: int) -> list[float]:
        """Compute EMA for a list of close prices."""
        if len(closes) < period:
            return closes
        k = 2.0 / (period + 1)
        ema = [sum(closes[:period]) / period]
        for price in closes[period:]:
            ema.append(price * k + ema[-1] * (1 - k))
        return ema
