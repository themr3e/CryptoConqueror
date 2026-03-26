"""Liquidity Sweep strategy.

Detects when price sweeps above a swing high (or below a swing low) and
reverses — a classic stop-hunt / liquidity grab setup.

Signal logic:
    BUY  — price wicks below recent swing low, closes back above it (sweep of
           sell-side liquidity), confirmed by bullish confirmation candle(s).
    SELL — price wicks above recent swing high, closes back below it (sweep of
           buy-side liquidity), confirmed by bearish confirmation candle(s).
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
from app.strategies.helpers.indicators import compute_atr
from app.strategies.helpers.swing_detection import find_swing_highs, find_swing_lows


class LiquiditySweepStrategy(BaseStrategy):
    """Identifies liquidity sweeps at swing highs/lows."""

    NAME = "liquidity_sweep"
    DEFAULT_PARAMS: dict[str, float] = {
        "SWING_ORDER": 5,
        "LOOKBACK": 50,
        "SL_ATR_MULT": 0.5,
        "TP1_RR": 1.5,
        "CONFIRM_BARS": 3,
    }

    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        """Generate liquidity sweep signals."""
        min_bars = int(self.params["LOOKBACK"]) + int(self.params["SWING_ORDER"]) + 20
        if len(candles) < min_bars:
            logger.debug("LiquiditySweep: insufficient candles ({})", len(candles))
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

        lookback = int(self.params["LOOKBACK"])
        order = int(self.params["SWING_ORDER"])
        sl_mult = float(self.params["SL_ATR_MULT"])
        tp1_rr = float(self.params["TP1_RR"])
        confirm_bars = int(self.params["CONFIRM_BARS"])

        # Work on recent window
        window = candles.iloc[-(lookback + order + confirm_bars + 5):]
        w_highs = window["high"].astype(float)
        w_lows = window["low"].astype(float)
        w_closes = window["close"].astype(float)
        w_opens = window["open"].astype(float)

        swing_highs_mask = find_swing_highs(w_highs, order=order)
        swing_lows_mask = find_swing_lows(w_lows, order=order)

        swing_high_vals = w_highs[swing_highs_mask]
        swing_low_vals = w_lows[swing_lows_mask]

        signals: list[CandidateSignal] = []

        # --- SELL signal: sweep above swing high then reverse ---
        if not swing_high_vals.empty:
            recent_sh = float(swing_high_vals.iloc[-1])
            sh_idx = swing_high_vals.index[-1]
            sh_pos = window.index.get_loc(sh_idx)

            # Need at least confirm_bars after the swing high
            after_sh = window.iloc[sh_pos + 1:]
            if len(after_sh) >= confirm_bars:
                # Check if price swept above swing high
                swept = after_sh["high"].astype(float).max() > recent_sh
                if swept:
                    # Confirm: last candle(s) close back below swing high
                    confirm_slice = after_sh.iloc[-confirm_bars:]
                    closes_below = (confirm_slice["close"].astype(float) < recent_sh).all()
                    last_bearish = float(w_closes.iloc[-1]) < float(w_opens.iloc[-1])

                    if closes_below and last_bearish:
                        entry = float(w_closes.iloc[-1])
                        sl = entry + sl_mult * current_atr
                        risk = sl - entry
                        if risk > 0:
                            tp1 = entry - tp1_rr * risk
                            tp2 = entry - (tp1_rr + 1.0) * risk
                            rr = abs(entry - tp1) / risk
                            confidence = self._calc_confidence(rr, current_atr)

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
                                confidence=self._to_decimal(min(confidence, 100.0), 2),
                                reasoning=(
                                    f"Liquidity sweep above swing high {recent_sh:.2f}; "
                                    f"bearish reversal confirmed over {confirm_bars} bars"
                                ),
                            ))

        # --- BUY signal: sweep below swing low then reverse ---
        if not swing_low_vals.empty:
            recent_sl = float(swing_low_vals.iloc[-1])
            sl_idx = swing_low_vals.index[-1]
            sl_pos = window.index.get_loc(sl_idx)

            after_sl = window.iloc[sl_pos + 1:]
            if len(after_sl) >= confirm_bars:
                swept = after_sl["low"].astype(float).min() < recent_sl
                if swept:
                    confirm_slice = after_sl.iloc[-confirm_bars:]
                    closes_above = (confirm_slice["close"].astype(float) > recent_sl).all()
                    last_bullish = float(w_closes.iloc[-1]) > float(w_opens.iloc[-1])

                    if closes_above and last_bullish:
                        entry = float(w_closes.iloc[-1])
                        sl = entry - sl_mult * current_atr
                        risk = entry - sl
                        if risk > 0:
                            tp1 = entry + tp1_rr * risk
                            tp2 = entry + (tp1_rr + 1.0) * risk
                            rr = abs(tp1 - entry) / risk
                            confidence = self._calc_confidence(rr, current_atr)

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
                                confidence=self._to_decimal(min(confidence, 100.0), 2),
                                reasoning=(
                                    f"Liquidity sweep below swing low {recent_sl:.2f}; "
                                    f"bullish reversal confirmed over {confirm_bars} bars"
                                ),
                            ))

        return signals

    def _calc_confidence(self, rr: float, atr: float) -> float:
        """Base confidence from RR ratio."""
        base = min(rr / 3.0, 1.0) * 60.0  # 0–60
        atr_bonus = min(atr / 5.0, 1.0) * 20.0  # 0–20 (higher ATR = more momentum)
        return 45.0 + base * 0.5 + atr_bonus * 0.25
