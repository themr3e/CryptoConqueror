"""EMA Momentum strategy.

Uses a triple EMA stack (fast, mid, slow=200) to confirm trend direction,
then enters on a strong momentum candle with body > BODY_ATR_MULT * ATR.

Signal logic:
    BUY  — EMA_FAST > EMA_MID > EMA_200, strong bullish body candle,
           entry above SWING_LOOKBACK swing high.
    SELL — EMA_FAST < EMA_MID < EMA_200, strong bearish body candle,
           entry below SWING_LOOKBACK swing low.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
from app.strategies.helpers.indicators import compute_atr, compute_ema
from app.strategies.helpers.swing_detection import get_recent_swing_high, get_recent_swing_low


class EMAMomentumStrategy(BaseStrategy):
    """Triple EMA stack + momentum candle entry."""

    NAME = "ema_momentum"
    DEFAULT_PARAMS: dict[str, float] = {
        "EMA_FAST": 20,
        "EMA_MID": 50,
        "BODY_ATR_MULT": 0.6,
        "SL_ATR_MULT": 1.0,
        "TP1_RR": 1.5,
        "SWING_LOOKBACK": 20,
    }

    _EMA_SLOW: int = 200

    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        """Generate EMA momentum signals."""
        min_bars = self._EMA_SLOW + 20
        if len(candles) < min_bars:
            logger.debug("EMAMomentum: insufficient candles ({})", len(candles))
            return []

        opens = candles["open"].astype(float)
        highs = candles["high"].astype(float)
        lows = candles["low"].astype(float)
        closes = candles["close"].astype(float)

        atr_series = compute_atr(highs, lows, closes, length=14)
        if atr_series.dropna().empty:
            return []

        current_atr = float(atr_series.dropna().iloc[-1])
        if current_atr <= 0:
            return []

        fast_period = int(self.params["EMA_FAST"])
        mid_period = int(self.params["EMA_MID"])
        body_mult = float(self.params["BODY_ATR_MULT"])
        sl_mult = float(self.params["SL_ATR_MULT"])
        tp1_rr = float(self.params["TP1_RR"])
        swing_lookback = int(self.params["SWING_LOOKBACK"])

        ema_fast = compute_ema(closes, fast_period)
        ema_mid = compute_ema(closes, mid_period)
        ema_slow = compute_ema(closes, self._EMA_SLOW)

        if any(s.dropna().empty for s in [ema_fast, ema_mid, ema_slow]):
            return []

        last_fast = float(ema_fast.iloc[-1])
        last_mid = float(ema_mid.iloc[-1])
        last_slow = float(ema_slow.iloc[-1])

        last_open = float(opens.iloc[-1])
        last_close = float(closes.iloc[-1])
        body_size = abs(last_close - last_open)
        strong_body = body_size >= body_mult * current_atr

        if not strong_body:
            return []

        signals: list[CandidateSignal] = []

        # --- BUY: bullish EMA stack + bullish momentum candle ---
        bullish_stack = last_fast > last_mid > last_slow
        bullish_candle = last_close > last_open

        if bullish_stack and bullish_candle:
            swing_high = get_recent_swing_high(highs, order=3, lookback=swing_lookback)
            recent_swing_str = f"{swing_high:.2f}" if swing_high else "N/A"

            entry = last_close
            sl = entry - sl_mult * current_atr
            risk = entry - sl
            if risk > 0:
                tp1 = entry + tp1_rr * risk
                tp2 = entry + (tp1_rr + 1.0) * risk
                rr = (tp1 - entry) / risk
                confidence = self._calc_confidence(rr, True, body_size / current_atr)

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
                    confidence=self._to_decimal(min(confidence, 95.0), 2),
                    reasoning=(
                        f"EMA momentum BUY: {fast_period}/{mid_period}/200 bullish stack "
                        f"({last_fast:.1f}>{last_mid:.1f}>{last_slow:.1f}), "
                        f"body={body_size:.2f} ({body_size/current_atr:.1f}x ATR), "
                        f"swing_high={recent_swing_str}"
                    ),
                ))

        # --- SELL: bearish EMA stack + bearish momentum candle ---
        bearish_stack = last_fast < last_mid < last_slow
        bearish_candle = last_close < last_open

        if bearish_stack and bearish_candle:
            swing_low = get_recent_swing_low(lows, order=3, lookback=swing_lookback)
            recent_swing_str = f"{swing_low:.2f}" if swing_low else "N/A"

            entry = last_close
            sl = entry + sl_mult * current_atr
            risk = sl - entry
            if risk > 0:
                tp1 = entry - tp1_rr * risk
                tp2 = entry - (tp1_rr + 1.0) * risk
                rr = (entry - tp1) / risk
                confidence = self._calc_confidence(rr, True, body_size / current_atr)

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
                    confidence=self._to_decimal(min(confidence, 95.0), 2),
                    reasoning=(
                        f"EMA momentum SELL: {fast_period}/{mid_period}/200 bearish stack "
                        f"({last_fast:.1f}<{last_mid:.1f}<{last_slow:.1f}), "
                        f"body={body_size:.2f} ({body_size/current_atr:.1f}x ATR), "
                        f"swing_low={recent_swing_str}"
                    ),
                ))

        return signals

    def _calc_confidence(self, rr: float, strong_stack: bool, body_ratio: float) -> float:
        """Calculate signal confidence."""
        base = 50.0
        if strong_stack:
            base += 15.0
        body_bonus = min(body_ratio / 3.0, 1.0) * 15.0
        rr_bonus = min(rr / 3.0, 1.0) * 10.0
        return min(base + body_bonus + rr_bonus, 95.0)
