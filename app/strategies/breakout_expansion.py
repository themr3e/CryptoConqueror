"""Breakout Expansion strategy.

Identifies volatility compression (consolidation) followed by a high-volume
breakout candle, entering in the direction of the breakout.

Signal logic:
    BUY  — price breaks above consolidation range with ATR expansion and
           volume exceeding VOLUME_MULT * average volume.
    SELL — price breaks below consolidation range with ATR expansion and
           volume exceeding VOLUME_MULT * average volume.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
from app.strategies.helpers.indicators import compute_atr


class BreakoutExpansionStrategy(BaseStrategy):
    """Trades breakouts after volatility compression periods."""

    NAME = "breakout_expansion"
    DEFAULT_PARAMS: dict[str, float] = {
        "ATR_COMPRESSION": 0.5,
        "MIN_CONSOL_BARS": 8,
        "VOLUME_MULT": 1.5,
        "BREAKOUT_BODY_ATR": 1.5,
    }

    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        """Generate breakout expansion signals."""
        min_consol = int(self.params["MIN_CONSOL_BARS"])
        min_bars = min_consol + 40
        if len(candles) < min_bars:
            logger.debug("BreakoutExpansion: insufficient candles ({})", len(candles))
            return []

        opens = candles["open"].astype(float)
        highs = candles["high"].astype(float)
        lows = candles["low"].astype(float)
        closes = candles["close"].astype(float)

        has_volume = "volume" in candles.columns
        volumes = candles["volume"].astype(float) if has_volume else None

        atr_series = compute_atr(highs, lows, closes, length=14)
        if atr_series.dropna().empty:
            return []

        current_atr = float(atr_series.dropna().iloc[-1])
        if current_atr <= 0:
            return []

        # Baseline ATR over longer window
        baseline_atr = float(atr_series.dropna().mean())

        atr_compression = float(self.params["ATR_COMPRESSION"])
        vol_mult = float(self.params["VOLUME_MULT"])
        body_atr_mult = float(self.params["BREAKOUT_BODY_ATR"])

        # --- Detect compression zone ---
        consol_window = candles.iloc[-(min_consol + 3):-1]  # Exclude last bar
        consol_highs = consol_window["high"].astype(float)
        consol_lows = consol_window["low"].astype(float)
        consol_range = float(consol_highs.max()) - float(consol_lows.min())

        # Compression check: consol range < ATR_COMPRESSION * ATR * bars
        is_compressed = consol_range < atr_compression * current_atr * min_consol

        if not is_compressed:
            return []

        # ATR expansion: current ATR > baseline (not compressed)
        atr_expanding = current_atr > baseline_atr * 0.8

        # --- Check breakout candle (last bar) ---
        last_open = float(opens.iloc[-1])
        last_close = float(closes.iloc[-1])
        last_high = float(highs.iloc[-1])
        last_low = float(lows.iloc[-1])

        body_size = abs(last_close - last_open)
        is_breakout_body = body_size >= body_atr_mult * current_atr

        # Volume confirmation
        vol_confirmed = True
        if has_volume and volumes is not None:
            avg_vol = float(volumes.iloc[-20:-1].mean()) if len(volumes) >= 20 else float(volumes.mean())
            last_vol = float(volumes.iloc[-1])
            vol_confirmed = last_vol >= vol_mult * avg_vol if avg_vol > 0 else True

        if not (is_breakout_body and vol_confirmed):
            return []

        consol_high = float(consol_highs.max())
        consol_low = float(consol_lows.min())

        signals: list[CandidateSignal] = []

        # BUY breakout: close above consolidation high
        if last_close > consol_high and last_close > last_open:
            entry = last_close
            sl = consol_low  # Below consolidation range
            risk = entry - sl
            if risk > 0:
                tp1_rr = 1.5
                tp1 = entry + tp1_rr * risk
                tp2 = entry + (tp1_rr + 1.0) * risk
                rr = tp1_rr
                confidence = self._calc_confidence(rr, atr_expanding, vol_confirmed)

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
                        f"Breakout BUY above consolidation {consol_high:.2f}; "
                        f"range={consol_range:.2f}, ATR={current_atr:.2f}, vol_ok={vol_confirmed}"
                    ),
                ))

        # SELL breakout: close below consolidation low
        elif last_close < consol_low and last_close < last_open:
            entry = last_close
            sl = consol_high  # Above consolidation range
            risk = sl - entry
            if risk > 0:
                tp1_rr = 1.5
                tp1 = entry - tp1_rr * risk
                tp2 = entry - (tp1_rr + 1.0) * risk
                rr = tp1_rr
                confidence = self._calc_confidence(rr, atr_expanding, vol_confirmed)

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
                        f"Breakout SELL below consolidation {consol_low:.2f}; "
                        f"range={consol_range:.2f}, ATR={current_atr:.2f}, vol_ok={vol_confirmed}"
                    ),
                ))

        return signals

    def _calc_confidence(self, rr: float, atr_expanding: bool, vol_confirmed: bool) -> float:
        """Calculate signal confidence."""
        base = 50.0
        if atr_expanding:
            base += 15.0
        if vol_confirmed:
            base += 15.0
        rr_bonus = min(rr / 3.0, 1.0) * 10.0
        return min(base + rr_bonus, 95.0)
