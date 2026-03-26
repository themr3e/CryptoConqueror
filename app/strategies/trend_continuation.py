"""Trend Continuation strategy.

Enters on pullbacks within an established EMA trend.

Signal logic:
    BUY  — price above EMA_FAST, pulls back within PULLBACK_ATR_MULT * ATR of
           the EMA, then prints a bullish candle.
    SELL — price below EMA_FAST, rallies within PULLBACK_ATR_MULT * ATR of
           the EMA, then prints a bearish candle.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
from app.strategies.helpers.indicators import compute_atr, compute_ema


class TrendContinuationStrategy(BaseStrategy):
    """Trades pullbacks to EMA within an established trend."""

    NAME = "trend_continuation"
    DEFAULT_PARAMS: dict[str, float] = {
        "EMA_FAST": 50,
        "PULLBACK_ATR_MULT": 1.0,
        "SL_ATR_MULT": 1.0,
        "TP1_RR": 2.0,
        "LOOKBACK_PULLBACK": 5,
    }

    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        """Generate trend continuation signals."""
        ema_period = int(self.params["EMA_FAST"])
        min_bars = ema_period + 30
        if len(candles) < min_bars:
            logger.debug("TrendContinuation: insufficient candles ({})", len(candles))
            return []

        opens = candles["open"].astype(float)
        highs = candles["high"].astype(float)
        lows = candles["low"].astype(float)
        closes = candles["close"].astype(float)

        atr_series = compute_atr(highs, lows, closes, length=14)
        ema_series = compute_ema(closes, ema_period)

        if atr_series.dropna().empty or ema_series.dropna().empty:
            return []

        current_atr = float(atr_series.dropna().iloc[-1])
        current_ema = float(ema_series.iloc[-1])
        current_close = float(closes.iloc[-1])
        current_open = float(opens.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) > 1 else current_close

        if current_atr <= 0:
            return []

        pullback_mult = float(self.params["PULLBACK_ATR_MULT"])
        sl_mult = float(self.params["SL_ATR_MULT"])
        tp1_rr = float(self.params["TP1_RR"])
        lookback_pb = int(self.params["LOOKBACK_PULLBACK"])

        signals: list[CandidateSignal] = []

        # Determine trend direction: EMA slope
        ema_lookback = min(lookback_pb, len(ema_series) - 1)
        ema_prev = float(ema_series.iloc[-(ema_lookback + 1)])
        ema_trending_up = current_ema > ema_prev
        ema_trending_down = current_ema < ema_prev

        # --- BUY: uptrend, price pulled back near EMA ---
        if ema_trending_up and current_close > current_ema:
            dist_to_ema = current_close - current_ema
            in_pullback_zone = dist_to_ema <= pullback_mult * current_atr

            # Recent candle touched/approached EMA
            recent_lows = lows.iloc[-lookback_pb:]
            touched_ema = any(
                float(l) <= current_ema + pullback_mult * current_atr
                for l in recent_lows
            )

            # Bullish confirmation candle
            bullish_candle = current_close > current_open

            if (in_pullback_zone or touched_ema) and bullish_candle:
                entry = current_close
                sl = entry - sl_mult * current_atr
                risk = entry - sl
                if risk > 0:
                    tp1 = entry + tp1_rr * risk
                    tp2 = entry + (tp1_rr + 1.0) * risk
                    rr = (tp1 - entry) / risk

                    signals.append(CandidateSignal(
                        strategy_name=self.NAME,
                        symbol="XAUUSD",
                        timeframe="H1",
                        direction=SignalDirection.BUY,
                        entry_price=self._to_decimal(entry),
                        stop_loss=self._to_decimal(sl),
                        take_profit_1=self._to_decimal(tp1),
                        take_profit_2=self._to_decimal(tp2),
                        risk_reward=self._to_decimal(rr, 2),
                        confidence=self._to_decimal(self._calc_confidence(rr, ema_trending_up, in_pullback_zone), 2),
                        reasoning=(
                            f"Trend continuation BUY: price {entry:.2f} near EMA{ema_period} "
                            f"{current_ema:.2f}, bullish reversal candle"
                        ),
                    ))

        # --- SELL: downtrend, price rallied near EMA ---
        elif ema_trending_down and current_close < current_ema:
            dist_to_ema = current_ema - current_close
            in_pullback_zone = dist_to_ema <= pullback_mult * current_atr

            recent_highs = highs.iloc[-lookback_pb:]
            touched_ema = any(
                float(h) >= current_ema - pullback_mult * current_atr
                for h in recent_highs
            )

            bearish_candle = current_close < current_open

            if (in_pullback_zone or touched_ema) and bearish_candle:
                entry = current_close
                sl = entry + sl_mult * current_atr
                risk = sl - entry
                if risk > 0:
                    tp1 = entry - tp1_rr * risk
                    tp2 = entry - (tp1_rr + 1.0) * risk
                    rr = (entry - tp1) / risk

                    signals.append(CandidateSignal(
                        strategy_name=self.NAME,
                        symbol="XAUUSD",
                        timeframe="H1",
                        direction=SignalDirection.SELL,
                        entry_price=self._to_decimal(entry),
                        stop_loss=self._to_decimal(sl),
                        take_profit_1=self._to_decimal(tp1),
                        take_profit_2=self._to_decimal(tp2),
                        risk_reward=self._to_decimal(rr, 2),
                        confidence=self._to_decimal(self._calc_confidence(rr, ema_trending_down, in_pullback_zone), 2),
                        reasoning=(
                            f"Trend continuation SELL: price {entry:.2f} near EMA{ema_period} "
                            f"{current_ema:.2f}, bearish reversal candle"
                        ),
                    ))

        return signals

    def _calc_confidence(self, rr: float, strong_trend: bool, in_zone: bool) -> float:
        """Calculate signal confidence."""
        base = 50.0
        if strong_trend:
            base += 15.0
        if in_zone:
            base += 10.0
        rr_bonus = min(rr / 3.0, 1.0) * 15.0
        return min(base + rr_bonus, 95.0)
