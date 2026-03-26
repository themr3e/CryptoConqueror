"""Crypto Breakout strategy.

Identifies a consolidation range over the past ``RANGE_PERIOD`` H1 candles,
then enters when price breaks cleanly outside the range with a confirming body.

Tuned for crypto: wider stops, higher RR targets, no session filter.

Signal logic:
    BUY  — close breaks above range_high by at least ``BREAKOUT_ATR_MULT * ATR``.
    SELL — close breaks below range_low by at least ``BREAKOUT_ATR_MULT * ATR``.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
from app.strategies.helpers.indicators import compute_atr


class CryptoBreakoutStrategy(BaseStrategy):
    """Range breakout for crypto futures with ATR confirmation."""

    NAME = "crypto_breakout"
    ASSET_CLASS = "crypto_futures"
    DEFAULT_PARAMS: dict[str, float] = {
        "RANGE_PERIOD": 24,         # look-back bars to define range (~1 day on H1)
        "BREAKOUT_ATR_MULT": 0.3,   # price must breach range by this * ATR
        "ATR_PERIOD": 14,
        "ATR_SL_MULT": 1.0,
        "TP1_RR": 1.5,
        "TP2_RR": 3.0,
        "MIN_CANDLES": 80,
    }

    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        """Generate crypto breakout signals."""
        min_bars = int(self.params["MIN_CANDLES"])
        if len(candles) < min_bars:
            logger.debug("CryptoBreakout: insufficient candles ({})", len(candles))
            return []

        opens = candles["open"].astype(float)
        highs = candles["high"].astype(float)
        lows = candles["low"].astype(float)
        closes = candles["close"].astype(float)

        range_period = int(self.params["RANGE_PERIOD"])
        breakout_mult = float(self.params["BREAKOUT_ATR_MULT"])
        atr_period = int(self.params["ATR_PERIOD"])
        sl_mult = float(self.params["ATR_SL_MULT"])
        tp1_rr = float(self.params["TP1_RR"])
        tp2_rr = float(self.params["TP2_RR"])

        atr_series = compute_atr(highs, lows, closes, length=atr_period)
        if atr_series.dropna().empty:
            return []

        current_atr = float(atr_series.dropna().iloc[-1])
        if current_atr <= 0:
            return []

        # Define range using bars BEFORE the current candle
        range_slice = candles.iloc[-(range_period + 1):-1]
        if len(range_slice) < range_period // 2:
            return []

        range_high = float(range_slice["high"].astype(float).max())
        range_low = float(range_slice["low"].astype(float).min())
        range_size = range_high - range_low

        # Range must be somewhat tight (consolidated) — less than 3% of price
        last_close = float(closes.iloc[-1])
        last_open = float(opens.iloc[-1])
        if range_size / last_close > 0.03:
            return []

        min_breakout_distance = breakout_mult * current_atr
        signals: list[CandidateSignal] = []
        symbol = self._infer_symbol(candles)

        # --- BUY breakout ---
        bullish_body = last_close > last_open
        if bullish_body and last_close > range_high + min_breakout_distance:
            entry = last_close
            sl = entry - sl_mult * current_atr
            risk = entry - sl
            if risk > 0:
                tp1 = entry + tp1_rr * risk
                tp2 = entry + tp2_rr * risk
                confidence = self._calc_confidence(range_size, current_atr, last_close)

                signals.append(CandidateSignal(
                    strategy_name=self.NAME,
                    symbol=symbol,
                    timeframe="H1",
                    direction=SignalDirection.BUY,
                    entry_price=self._to_decimal(entry),
                    stop_loss=self._to_decimal(sl),
                    take_profit_1=self._to_decimal(tp1),
                    take_profit_2=self._to_decimal(tp2),
                    risk_reward=self._to_decimal(tp1_rr, 2),
                    confidence=self._to_decimal(min(confidence, 90.0), 2),
                    reasoning=(
                        f"Crypto breakout BUY above range_high {range_high:.2f} "
                        f"(range={range_size:.2f}, ATR={current_atr:.2f})"
                    ),
                    session="24h",
                ))

        # --- SELL breakout ---
        bearish_body = last_close < last_open
        if bearish_body and last_close < range_low - min_breakout_distance:
            entry = last_close
            sl = entry + sl_mult * current_atr
            risk = sl - entry
            if risk > 0:
                tp1 = entry - tp1_rr * risk
                tp2 = entry - tp2_rr * risk
                confidence = self._calc_confidence(range_size, current_atr, last_close)

                signals.append(CandidateSignal(
                    strategy_name=self.NAME,
                    symbol=symbol,
                    timeframe="H1",
                    direction=SignalDirection.SELL,
                    entry_price=self._to_decimal(entry),
                    stop_loss=self._to_decimal(sl),
                    take_profit_1=self._to_decimal(tp1),
                    take_profit_2=self._to_decimal(tp2),
                    risk_reward=self._to_decimal(tp1_rr, 2),
                    confidence=self._to_decimal(min(confidence, 90.0), 2),
                    reasoning=(
                        f"Crypto breakout SELL below range_low {range_low:.2f} "
                        f"(range={range_size:.2f}, ATR={current_atr:.2f})"
                    ),
                    session="24h",
                ))

        return signals

    def _infer_symbol(self, candles: pd.DataFrame) -> str:
        if hasattr(candles, "attrs") and "symbol" in candles.attrs:
            return candles.attrs["symbol"]
        return "BTCUSDT"

    def _calc_confidence(self, range_size: float, atr: float, price: float) -> float:
        """Confidence is higher when range is tight relative to ATR."""
        base = 50.0
        range_atr_ratio = range_size / atr if atr > 0 else 10.0
        # Tight range = high confidence breakout
        if range_atr_ratio < 3:
            base += 20.0
        elif range_atr_ratio < 6:
            base += 10.0
        return min(base, 90.0)
