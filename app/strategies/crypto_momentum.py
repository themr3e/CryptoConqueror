"""Crypto Momentum strategy.

Uses a dual EMA crossover (EMA_FAST / EMA_SLOW) with volume confirmation to
enter in the direction of momentum on BTC/ETH perpetual futures.

Calibrated for crypto volatility:
  - Wider ATR-based stops (1.5x) to survive noise.
  - High TP2 RR (4.0) to capture crypto swing extensions.
  - Volume confirmation: candle volume must exceed 1.5x the 20-bar average.
  - No session filter — crypto trades 24/7.

Signal logic:
    BUY  — EMA_FAST crosses above EMA_SLOW on close, volume confirmed.
    SELL — EMA_FAST crosses below EMA_SLOW on close, volume confirmed.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
from app.strategies.helpers.indicators import compute_atr, compute_ema


class CryptoMomentumStrategy(BaseStrategy):
    """EMA crossover + volume confirmation for crypto futures."""

    NAME = "crypto_momentum"
    ASSET_CLASS = "crypto_futures"
    DEFAULT_PARAMS: dict[str, float] = {
        "EMA_FAST": 21,
        "EMA_SLOW": 55,
        "ATR_PERIOD": 14,
        "ATR_SL_MULT": 1.5,
        "TP1_RR": 2.0,
        "TP2_RR": 4.0,
        "VOLUME_CONFIRM_MULT": 1.5,
        "MIN_CANDLES": 100,
    }

    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        """Generate crypto momentum signals."""
        min_bars = int(self.params["MIN_CANDLES"])
        if len(candles) < min_bars:
            logger.debug("CryptoMomentum: insufficient candles ({})", len(candles))
            return []

        opens = candles["open"].astype(float)
        highs = candles["high"].astype(float)
        lows = candles["low"].astype(float)
        closes = candles["close"].astype(float)

        has_volume = "volume" in candles.columns
        volumes = candles["volume"].astype(float) if has_volume else None

        fast_period = int(self.params["EMA_FAST"])
        slow_period = int(self.params["EMA_SLOW"])
        atr_period = int(self.params["ATR_PERIOD"])
        sl_mult = float(self.params["ATR_SL_MULT"])
        tp1_rr = float(self.params["TP1_RR"])
        tp2_rr = float(self.params["TP2_RR"])
        vol_mult = float(self.params["VOLUME_CONFIRM_MULT"])

        ema_fast = compute_ema(closes, fast_period)
        ema_slow = compute_ema(closes, slow_period)
        atr_series = compute_atr(highs, lows, closes, length=atr_period)

        if atr_series.dropna().empty:
            return []

        current_atr = float(atr_series.dropna().iloc[-1])
        if current_atr <= 0:
            return []

        # Get last two bars for crossover detection
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return []

        fast_now = float(ema_fast.iloc[-1])
        fast_prev = float(ema_fast.iloc[-2])
        slow_now = float(ema_slow.iloc[-1])
        slow_prev = float(ema_slow.iloc[-2])

        # Volume confirmation
        vol_ok = True
        if has_volume and volumes is not None:
            avg_vol = float(volumes.iloc[-21:-1].mean()) if len(volumes) >= 21 else float(volumes.mean())
            last_vol = float(volumes.iloc[-1])
            vol_ok = last_vol >= vol_mult * avg_vol if avg_vol > 0 else True

        entry = float(closes.iloc[-1])
        signals: list[CandidateSignal] = []

        # --- BUY crossover: fast crossed above slow ---
        bullish_cross = fast_prev <= slow_prev and fast_now > slow_now
        if bullish_cross and vol_ok:
            sl = entry - sl_mult * current_atr
            risk = entry - sl
            if risk > 0:
                tp1 = entry + tp1_rr * risk
                tp2 = entry + tp2_rr * risk

                symbol = self._infer_symbol(candles)
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
                    confidence=self._to_decimal(self._calc_confidence(vol_ok, current_atr, entry), 2),
                    reasoning=(
                        f"Crypto momentum BUY: EMA{fast_period} ({fast_now:.2f}) crossed above "
                        f"EMA{slow_period} ({slow_now:.2f}), vol_ok={vol_ok}, ATR={current_atr:.2f}"
                    ),
                    session="24h",
                ))

        # --- SELL crossover: fast crossed below slow ---
        bearish_cross = fast_prev >= slow_prev and fast_now < slow_now
        if bearish_cross and vol_ok:
            sl = entry + sl_mult * current_atr
            risk = sl - entry
            if risk > 0:
                tp1 = entry - tp1_rr * risk
                tp2 = entry - tp2_rr * risk

                symbol = self._infer_symbol(candles)
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
                    confidence=self._to_decimal(self._calc_confidence(vol_ok, current_atr, entry), 2),
                    reasoning=(
                        f"Crypto momentum SELL: EMA{fast_period} ({fast_now:.2f}) crossed below "
                        f"EMA{slow_period} ({slow_now:.2f}), vol_ok={vol_ok}, ATR={current_atr:.2f}"
                    ),
                    session="24h",
                ))

        return signals

    def _infer_symbol(self, candles: pd.DataFrame) -> str:
        """Try to read symbol from DataFrame metadata, fall back to BTCUSDT."""
        if hasattr(candles, "attrs") and "symbol" in candles.attrs:
            return candles.attrs["symbol"]
        return "BTCUSDT"

    def _calc_confidence(self, vol_ok: bool, atr: float, price: float) -> float:
        """Calculate signal confidence (50–90 range)."""
        base = 55.0
        if vol_ok:
            base += 15.0
        # ATR as % of price gives volatility context
        atr_pct = atr / price if price > 0 else 0
        # Favour moderate volatility (0.5-2% ATR/price range)
        if 0.005 <= atr_pct <= 0.02:
            base += 10.0
        return min(base, 90.0)
