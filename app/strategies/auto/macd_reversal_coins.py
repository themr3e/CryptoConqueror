"""MACD Divergence Reversal strategies — one per coin.

Auto-loaded by app/strategies/__init__.py at startup.

Four dedicated strategies, each with params optimised on 1500 H1 bars:
  zec_macd_reversal  — ZECUSDT  WR=66.2% P&L=+29.2%  (div_lookback=20, swing_gap=4, RR=0.7)
  btc_macd_reversal  — BTCUSDT  WR=65.6% P&L=+9.5%   (div_lookback=40, swing_gap=2, RR=1.3)
  bch_macd_reversal  — BCHUSDT  WR=70.6% P&L=+131.9% (div_lookback=40, swing_gap=3, RR=1.5)
  eth_macd_reversal  — ETHUSDT  WR=85.3% P&L=+109.2% (div_lookback=15, swing_gap=2, RR=2.0)

Strategy logic (MACD Divergence Reversal):
  - Bullish divergence: price makes lower low, MACD makes higher low → BUY
  - Bearish divergence: price makes higher high, MACD makes lower high → SELL
  - Histogram trigger: 2 consecutive shrinking histogram bars (exhaustion)
  - RSI filter: not overbought on BUY, not oversold on SELL
  - SL: ATR × atr_sl_mult
  - TP1: SL × rr  (main target)
  - TP2: SL × rr × 1.5  (extension target — optional, for bot's TP2 field)

Confidence model:
  Base = 60%
  +10% if histogram trigger fires strongly (3 bars shrinking)
  +10% if RSI is in reversal zone (BUY: RSI < 40, SELL: RSI > 60)
  +5%  if divergence is sharp (price move > 2× MACD move)
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
from app.strategies.helpers.indicators import compute_atr


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _macd(close: pd.Series, fast: int, slow: int, sig: int):
    ef   = close.ewm(span=fast, adjust=False).mean()
    es   = close.ewm(span=slow, adjust=False).mean()
    macd = (ef - es).to_numpy(dtype=float)
    line = pd.Series(macd).ewm(span=sig, adjust=False).mean().to_numpy(dtype=float)
    hist = macd - line
    return macd, line, hist


def _rsi(close: pd.Series, period: int = 14) -> np.ndarray:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    return (100 - 100 / (1 + gain / loss.replace(0, 1e-9))).to_numpy(dtype=float)


def _swing_highs(arr: np.ndarray, lookback: int, gap: int) -> list[int]:
    """Indices of local maxima within the last `lookback` bars."""
    end = len(arr)
    start = max(0, end - lookback)
    result = []
    for i in range(start + gap, end - gap):
        if all(arr[i] >= arr[i - j] for j in range(1, gap + 1)) and \
           all(arr[i] >= arr[i + j] for j in range(1, gap + 1)):
            result.append(i)
    return result


def _swing_lows(arr: np.ndarray, lookback: int, gap: int) -> list[int]:
    """Indices of local minima within the last `lookback` bars."""
    end = len(arr)
    start = max(0, end - lookback)
    result = []
    for i in range(start + gap, end - gap):
        if all(arr[i] <= arr[i - j] for j in range(1, gap + 1)) and \
           all(arr[i] <= arr[i + j] for j in range(1, gap + 1)):
            result.append(i)
    return result


def _hist_trigger(hist: np.ndarray, trigger_bars: int, direction: str) -> tuple[bool, int]:
    """Check for histogram exhaustion: N consecutive shrinking bars.
    Returns (triggered, strength) where strength = number of shrinking bars found.
    """
    n = len(hist)
    if n < trigger_bars + 1:
        return False, 0

    recent = hist[-(trigger_bars + 2):]
    if direction == "BUY":
        # Histogram is negative and getting less negative (rising toward zero)
        shrinking = all(recent[i] < recent[i - 1] for i in range(1, len(recent)))
        # Wait — for bullish we want histogram flipping from negative toward positive
        # Check: histogram was negative but recent bars are increasing (less negative or flipping)
        neg_count = sum(1 for x in recent if x < 0)
        rising = all(recent[i] > recent[i - 1] for i in range(-trigger_bars, 0))
        triggered = neg_count >= 2 and rising
        strength = sum(1 for i in range(-trigger_bars, 0) if recent[i] > recent[i - 1])
    else:
        # Histogram was positive but recent bars are decreasing (less positive or flipping)
        pos_count = sum(1 for x in recent if x > 0)
        falling = all(recent[i] < recent[i - 1] for i in range(-trigger_bars, 0))
        triggered = pos_count >= 2 and falling
        strength = sum(1 for i in range(-trigger_bars, 0) if recent[i] < recent[i - 1])

    return triggered, strength


def _confidence(rsi_val: float, direction: str, hist_strength: int, div_sharpness: float) -> float:
    base = 60.0
    if direction == "BUY":
        if rsi_val < 40:   base += 10.0
        elif rsi_val < 50: base += 5.0
    else:
        if rsi_val > 60:   base += 10.0
        elif rsi_val > 50: base += 5.0

    base += min(hist_strength * 3.0, 10.0)   # up to +10% for strong trigger
    if div_sharpness > 2.0:
        base += 5.0   # sharp divergence = higher confidence

    return min(base, 95.0)


# ── Base MACD Reversal ────────────────────────────────────────────────────────

class _MACDReversalBase(BaseStrategy):
    """MACD divergence reversal — base class for per-coin variants.
    NAME intentionally blank so this class is NOT auto-registered as a tradable strategy.
    """

    NAME = ""   # empty → skipped by __init_subclass__ registry
    ASSET_CLASS = "crypto_futures"

    DEFAULT_PARAMS: dict[str, float] = {
        "macd_fast":    12,
        "macd_slow":    26,
        "macd_signal":  9,
        "div_lookback": 20,
        "swing_gap":    2,
        "hist_trigger": 2,
        "rsi_buy_max":  78,
        "rsi_sell_min": 22,
        "atr_sl_mult":  2.0,
        "rr":           1.5,
        "min_candles":  120,
    }

    SYMBOL: str = "BTCUSDT"   # overridden per coin

    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        min_bars = int(self.params["min_candles"])
        if len(candles) < min_bars:
            return []

        close  = candles["close"].astype(float).reset_index(drop=True)
        high_s = candles["high"].astype(float).reset_index(drop=True)
        low_s  = candles["low"].astype(float).reset_index(drop=True)

        fast = int(self.params["macd_fast"])
        slow = int(self.params["macd_slow"])
        sig  = int(self.params["macd_signal"])
        lb   = int(self.params["div_lookback"])
        gap  = int(self.params["swing_gap"])
        ht   = int(self.params["hist_trigger"])
        rbm  = float(self.params["rsi_buy_max"])
        rsm  = float(self.params["rsi_sell_min"])
        slm  = float(self.params["atr_sl_mult"])
        rr   = float(self.params["rr"])

        macd_arr, _, hist_arr = _macd(close, fast, slow, sig)
        rsi_arr = _rsi(close)
        atr_s   = compute_atr(high_s, low_s, close, length=14)
        atr_arr = atr_s.to_numpy(dtype=float)

        closes = close.to_numpy(dtype=float)
        n = len(closes)
        entry  = closes[-1]
        atr_v  = atr_arr[-1] if not np.isnan(atr_arr[-1]) else 0
        if atr_v <= 0:
            return []

        rsi_v = rsi_arr[-1]
        sl_d  = atr_v * slm
        tp1_d = sl_d * rr
        tp2_d = sl_d * rr * 1.5

        signals: list[CandidateSignal] = []

        # ── Bullish divergence check ──────────────────────────────────────────
        if rsi_v <= rbm:
            price_lows = _swing_lows(closes, lb, gap)
            macd_lows  = _swing_lows(macd_arr, lb, gap)

            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                p1, p2 = price_lows[-2], price_lows[-1]
                m1, m2 = macd_lows[-2],  macd_lows[-1]

                # Bullish div: price lower low, MACD higher low
                if closes[p2] < closes[p1] and macd_arr[m2] > macd_arr[m1]:
                    triggered, strength = _hist_trigger(hist_arr, ht, "BUY")
                    if triggered:
                        price_range = abs(closes[p2] - closes[p1])
                        macd_range  = abs(macd_arr[m2] - macd_arr[m1])
                        sharpness   = (price_range / entry) / (macd_range + 1e-9)
                        conf = _confidence(rsi_v, "BUY", strength, sharpness)

                        signals.append(CandidateSignal(
                            strategy_name = self.NAME,
                            symbol        = self.SYMBOL,
                            timeframe     = "H1",
                            direction     = SignalDirection.BUY,
                            entry_price   = self._to_decimal(entry),
                            stop_loss     = self._to_decimal(entry - sl_d),
                            take_profit_1 = self._to_decimal(entry + tp1_d),
                            take_profit_2 = self._to_decimal(entry + tp2_d),
                            risk_reward   = self._to_decimal(rr),
                            confidence    = self._to_decimal(conf, 1),
                            reasoning     = (
                                f"Bullish div: price LL ({closes[p1]:.2f}→{closes[p2]:.2f}) "
                                f"MACD HL ({macd_arr[m1]:.4f}→{macd_arr[m2]:.4f}) "
                                f"RSI={rsi_v:.0f} hist_strength={strength}"
                            ),
                        ))
                        logger.info("{} BUY signal: conf={:.0f}% RSI={:.0f}", self.SYMBOL, conf, rsi_v)

        # ── Bearish divergence check ──────────────────────────────────────────
        if rsi_v >= rsm:
            price_highs = _swing_highs(closes, lb, gap)
            macd_highs  = _swing_highs(macd_arr, lb, gap)

            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                p1, p2 = price_highs[-2], price_highs[-1]
                m1, m2 = macd_highs[-2],  macd_highs[-1]

                # Bearish div: price higher high, MACD lower high
                if closes[p2] > closes[p1] and macd_arr[m2] < macd_arr[m1]:
                    triggered, strength = _hist_trigger(hist_arr, ht, "SELL")
                    if triggered:
                        price_range = abs(closes[p2] - closes[p1])
                        macd_range  = abs(macd_arr[m2] - macd_arr[m1])
                        sharpness   = (price_range / entry) / (macd_range + 1e-9)
                        conf = _confidence(rsi_v, "SELL", strength, sharpness)

                        signals.append(CandidateSignal(
                            strategy_name = self.NAME,
                            symbol        = self.SYMBOL,
                            timeframe     = "H1",
                            direction     = SignalDirection.SELL,
                            entry_price   = self._to_decimal(entry),
                            stop_loss     = self._to_decimal(entry + sl_d),
                            take_profit_1 = self._to_decimal(entry - tp1_d),
                            take_profit_2 = self._to_decimal(entry - tp2_d),
                            risk_reward   = self._to_decimal(rr),
                            confidence    = self._to_decimal(conf, 1),
                            reasoning     = (
                                f"Bearish div: price HH ({closes[p1]:.2f}→{closes[p2]:.2f}) "
                                f"MACD LH ({macd_arr[m1]:.4f}→{macd_arr[m2]:.4f}) "
                                f"RSI={rsi_v:.0f} hist_strength={strength}"
                            ),
                        ))
                        logger.info("{} SELL signal: conf={:.0f}% RSI={:.0f}", self.SYMBOL, conf, rsi_v)

        return signals


# ── Per-coin subclasses — params locked from optimizer run ────────────────────

class ZECMACDReversal(_MACDReversalBase):
    """ZECUSDT — WR=66.2% P&L=+29.2% (65 trades, 1500 H1 bars)."""
    NAME   = "zec_macd_reversal"
    SYMBOL = "ZECUSDT"
    DEFAULT_PARAMS = {
        **_MACDReversalBase.DEFAULT_PARAMS,
        "div_lookback": 20,
        "swing_gap":    4,
        "hist_trigger": 2,
        "rsi_buy_max":  68,
        "rsi_sell_min": 32,
        "atr_sl_mult":  2.5,
        "rr":           0.7,
        "min_candles":  120,
    }


class BTCMACDReversal(_MACDReversalBase):
    """BTCUSDT — WR=65.6% P&L=+9.5% (93 trades, 1500 H1 bars)."""
    NAME   = "btc_macd_reversal"
    SYMBOL = "BTCUSDT"
    DEFAULT_PARAMS = {
        **_MACDReversalBase.DEFAULT_PARAMS,
        "div_lookback": 40,
        "swing_gap":    2,
        "hist_trigger": 2,
        "rsi_buy_max":  68,
        "rsi_sell_min": 22,
        "atr_sl_mult":  1.0,
        "rr":           1.3,
        "min_candles":  120,
    }


class BCHMACDReversal(_MACDReversalBase):
    """BCHUSDT — WR=70.6% P&L=+131.9% (102 trades, 1500 H1 bars)."""
    NAME   = "bch_macd_reversal"
    SYMBOL = "BCHUSDT"
    DEFAULT_PARAMS = {
        **_MACDReversalBase.DEFAULT_PARAMS,
        "div_lookback": 40,
        "swing_gap":    3,
        "hist_trigger": 3,
        "rsi_buy_max":  72,
        "rsi_sell_min": 28,
        "atr_sl_mult":  2.5,
        "rr":           1.5,
        "min_candles":  120,
    }


class ETHMACDReversal(_MACDReversalBase):
    """ETHUSDT — WR=85.3% P&L=+109.2% (34 trades, 1500 H1 bars)."""
    NAME   = "eth_macd_reversal"
    SYMBOL = "ETHUSDT"
    DEFAULT_PARAMS = {
        **_MACDReversalBase.DEFAULT_PARAMS,
        "div_lookback": 15,
        "swing_gap":    2,
        "hist_trigger": 3,
        "rsi_buy_max":  68,
        "rsi_sell_min": 22,
        "atr_sl_mult":  2.5,
        "rr":           2.0,
        "min_candles":  120,
    }
