"""Crypto SLC (Structure + Level + Confirmation) strategy.

Implements the Data Traders SLC system extracted from the ICT/SMC methodology:
  1. Structure  — only trade in direction of the trend (HH/HL or LH/LL)
  2. Level      — identify a key untested Order Block (OB) or Fair Value Gap (FVG)
  3. Confirmation — first pullback into the level with rejection candle

Entry rules:
  - BUY  when structure is uptrend + price pulls back into bullish OB/FVG + rejection
  - SELL when structure is downtrend + price pulls into bearish OB/FVG + rejection
  - Sideways market → no trade

Risk rules:
  - SL below OB/FVG zone (bull) or above zone (bear) + 0.3 ATR buffer
  - TP = entry ± 2.0 * risk distance
  - Minimum RR 1.5 (reject tighter setups)
  - Confidence 70 base, +10 when OB and FVG overlap (confluence) → max 80
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection
from app.strategies.helpers.indicators import compute_atr
from app.strategies.helpers.market_structure import detect_higher_highs_higher_lows


# ── Internal zone dataclass ────────────────────────────────────────────────────

@dataclass
class _Zone:
    """Price zone (OB or FVG) with metadata."""
    kind: str          # "ob" or "fvg"
    direction: str     # "bullish" or "bearish"
    zone_low: float
    zone_high: float
    bar_index: int     # index in original DataFrame where zone was formed
    mitigated: bool = False


# ── Strategy class ─────────────────────────────────────────────────────────────

class CryptoSLCStrategy(BaseStrategy):
    """Structure + Level + Confirmation for crypto futures (H1 candles)."""

    NAME = "crypto_slc"
    ASSET_CLASS = "crypto_futures"
    DEFAULT_PARAMS: dict[str, float] = {
        "ATR_PERIOD": 14,
        "IMPULSIVE_ATR_MULT": 1.5,   # candle move >= mult * ATR → impulsive
        "SL_BUFFER_ATR": 0.3,        # extra ATR buffer beyond zone edge for SL
        "TP_RR": 2.0,                # risk-reward for take profit
        "MIN_RR": 1.5,               # reject setups with RR below this
        "LOOKBACK_BARS": 60,         # max bars to scan back for OB/FVG
        "STRUCTURE_LOOKBACK": 20,    # bars passed to detect_higher_highs_higher_lows
        "BASE_CONFIDENCE": 70.0,
        "CONFLUENCE_BONUS": 10.0,
        "MIN_CANDLES": 100,
    }

    # ── Public entry point ─────────────────────────────────────────────────────

    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        """Return at most one CandidateSignal (highest confidence) per call."""
        min_bars = int(self.params["MIN_CANDLES"])
        if len(candles) < min_bars:
            logger.debug("CryptoSLC: insufficient candles ({})", len(candles))
            return []

        opens  = candles["open"].astype(float)
        highs  = candles["high"].astype(float)
        lows   = candles["low"].astype(float)
        closes = candles["close"].astype(float)

        # ── 1. Structure ───────────────────────────────────────────────────────
        structure_lookback = int(self.params["STRUCTURE_LOOKBACK"])
        structure = detect_higher_highs_higher_lows(highs, lows, lookback=structure_lookback)
        if structure == "sideways":
            logger.debug("CryptoSLC: sideways market — skip")
            return []

        trend_direction = "bullish" if structure == "uptrend" else "bearish"

        # ── 2. ATR ─────────────────────────────────────────────────────────────
        atr_period = int(self.params["ATR_PERIOD"])
        atr_series = compute_atr(highs, lows, closes, length=atr_period)
        valid_atr = atr_series.dropna()
        if valid_atr.empty:
            return []
        current_atr = float(valid_atr.iloc[-1])
        if current_atr <= 0:
            return []

        # ── 3. Detect unmitigated levels ───────────────────────────────────────
        lookback = int(self.params["LOOKBACK_BARS"])
        impulsive_mult = float(self.params["IMPULSIVE_ATR_MULT"])

        ob_zones  = self._detect_order_blocks(opens, highs, lows, closes, atr_series,
                                               trend_direction, lookback, impulsive_mult)
        fvg_zones = self._detect_fvgs(highs, lows, trend_direction, lookback)

        # Filter to unmitigated zones and mark mitigation status
        ob_zones  = self._filter_unmitigated(ob_zones, closes, highs, lows)
        fvg_zones = self._filter_unmitigated(fvg_zones, closes, highs, lows)

        if not ob_zones and not fvg_zones:
            logger.debug("CryptoSLC: no unmitigated {} levels found", trend_direction)
            return []

        # ── 4. Check confirmation at current candle ────────────────────────────
        current_close  = float(closes.iloc[-1])
        current_open   = float(opens.iloc[-1])
        current_high   = float(highs.iloc[-1])
        current_low    = float(lows.iloc[-1])

        sl_buffer_atr  = float(self.params["SL_BUFFER_ATR"])
        tp_rr          = float(self.params["TP_RR"])
        min_rr         = float(self.params["MIN_RR"])
        base_conf      = float(self.params["BASE_CONFIDENCE"])
        conf_bonus     = float(self.params["CONFLUENCE_BONUS"])

        symbol = self._infer_symbol(candles)
        candidates: list[CandidateSignal] = []

        for zone in ob_zones + fvg_zones:
            # Price must be entering (or inside) the zone on this candle
            if not self._price_entering_zone(zone, current_open, current_close,
                                              current_high, current_low, trend_direction):
                continue

            # Rejection confirmation
            if not self._has_rejection(zone, current_open, current_close,
                                        current_high, current_low, trend_direction):
                continue

            # SL / TP calculation
            if trend_direction == "bullish":
                sl = zone.zone_low - sl_buffer_atr * current_atr
                risk = current_close - sl
            else:
                sl = zone.zone_high + sl_buffer_atr * current_atr
                risk = sl - current_close

            if risk <= 0:
                continue

            tp = current_close + tp_rr * risk if trend_direction == "bullish" \
                 else current_close - tp_rr * risk
            rr = abs(tp - current_close) / risk

            if rr < min_rr:
                continue

            # Confidence: check for confluence (OB + FVG overlap)
            confidence = base_conf
            if self._has_confluence(zone, ob_zones, fvg_zones):
                confidence = min(confidence + conf_bonus, 80.0)

            direction = SignalDirection.BUY if trend_direction == "bullish" else SignalDirection.SELL

            reasoning = (
                f"SLC {structure} | {zone.kind.upper()} {zone.direction} zone "
                f"[{zone.zone_low:.4f}-{zone.zone_high:.4f}] | "
                f"ATR={current_atr:.4f} | RR={rr:.2f} | conf={confidence:.0f}"
            )

            candidates.append(CandidateSignal(
                strategy_name=self.NAME,
                symbol=symbol,
                timeframe="H1",
                direction=direction,
                entry_price=self._to_decimal(current_close),
                stop_loss=self._to_decimal(sl),
                take_profit_1=self._to_decimal(tp),
                take_profit_2=None,
                risk_reward=self._to_decimal(round(rr, 2), 2),
                confidence=self._to_decimal(confidence, 2),
                reasoning=reasoning,
                session="24h",
            ))

        if not candidates:
            return []

        # Return only the highest-confidence signal
        best = max(candidates, key=lambda s: float(s.confidence))
        return [best]

    # ── Order Block detection ──────────────────────────────────────────────────

    def _detect_order_blocks(
        self,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        atr_series: pd.Series,
        direction: str,
        lookback: int,
        impulsive_mult: float,
    ) -> list[_Zone]:
        """Detect Order Blocks within the lookback window.

        Bullish OB: last bearish candle immediately before an impulsive up move.
        Bearish OB: last bullish candle immediately before an impulsive down move.
        Impulsive = next candle's move >= impulsive_mult * ATR.
        """
        zones: list[_Zone] = []
        n = len(closes)
        start = max(0, n - lookback - 1)

        for i in range(start, n - 1):
            atr_val = float(atr_series.iloc[i]) if i < len(atr_series) else float("nan")
            if atr_val != atr_val or atr_val <= 0:  # nan check
                continue

            next_move = abs(float(closes.iloc[i + 1]) - float(opens.iloc[i + 1]))

            if direction == "bullish":
                # Bullish OB: bearish candle (close < open) followed by impulsive up
                is_bearish_candle = float(closes.iloc[i]) < float(opens.iloc[i])
                is_impulsive_up = (
                    float(closes.iloc[i + 1]) > float(opens.iloc[i + 1])
                    and next_move >= impulsive_mult * atr_val
                )
                if is_bearish_candle and is_impulsive_up:
                    zones.append(_Zone(
                        kind="ob",
                        direction="bullish",
                        zone_low=float(lows.iloc[i]),
                        zone_high=float(highs.iloc[i]),
                        bar_index=i,
                    ))
            else:
                # Bearish OB: bullish candle (close > open) followed by impulsive down
                is_bullish_candle = float(closes.iloc[i]) > float(opens.iloc[i])
                is_impulsive_down = (
                    float(closes.iloc[i + 1]) < float(opens.iloc[i + 1])
                    and next_move >= impulsive_mult * atr_val
                )
                if is_bullish_candle and is_impulsive_down:
                    zones.append(_Zone(
                        kind="ob",
                        direction="bearish",
                        zone_low=float(lows.iloc[i]),
                        zone_high=float(highs.iloc[i]),
                        bar_index=i,
                    ))

        return zones

    # ── FVG detection ──────────────────────────────────────────────────────────

    def _detect_fvgs(
        self,
        highs: pd.Series,
        lows: pd.Series,
        direction: str,
        lookback: int,
    ) -> list[_Zone]:
        """Detect Fair Value Gaps within the lookback window.

        Bullish FVG: candle[i-2].high < candle[i].low  (gap upward)
        Bearish FVG: candle[i-2].low  > candle[i].high (gap downward)
        Zone = the gap itself.
        """
        zones: list[_Zone] = []
        n = len(highs)
        start = max(2, n - lookback)

        for i in range(start, n):
            h_prev2 = float(highs.iloc[i - 2])
            l_prev2 = float(lows.iloc[i - 2])
            h_curr  = float(highs.iloc[i])
            l_curr  = float(lows.iloc[i])

            if direction == "bullish" and h_prev2 < l_curr:
                zones.append(_Zone(
                    kind="fvg",
                    direction="bullish",
                    zone_low=h_prev2,
                    zone_high=l_curr,
                    bar_index=i,
                ))
            elif direction == "bearish" and l_prev2 > h_curr:
                zones.append(_Zone(
                    kind="fvg",
                    direction="bearish",
                    zone_low=h_curr,
                    zone_high=l_prev2,
                    bar_index=i,
                ))

        return zones

    # ── Mitigation filter ──────────────────────────────────────────────────────

    def _filter_unmitigated(
        self,
        zones: list[_Zone],
        closes: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
    ) -> list[_Zone]:
        """Keep only zones that have NOT been fully mitigated yet.

        A bullish zone is mitigated when price closes below zone_low after formation.
        A bearish zone is mitigated when price closes above zone_high after formation.
        The current (last) candle is excluded from mitigation check — we want to allow
        the current bar to be the first entry into the zone.
        """
        unmitigated: list[_Zone] = []
        n = len(closes)
        for zone in zones:
            bar_after = zone.bar_index + 1
            check_end = n - 1  # exclude current candle
            if bar_after >= check_end:
                # Zone formed on the candle just before current — trivially unmitigated
                unmitigated.append(zone)
                continue

            mitigated = False
            for j in range(bar_after, check_end):
                if zone.direction == "bullish" and float(closes.iloc[j]) < zone.zone_low:
                    mitigated = True
                    break
                if zone.direction == "bearish" and float(closes.iloc[j]) > zone.zone_high:
                    mitigated = True
                    break

            if not mitigated:
                unmitigated.append(zone)

        return unmitigated

    # ── Confirmation checks ────────────────────────────────────────────────────

    def _price_entering_zone(
        self,
        zone: _Zone,
        open_: float,
        close: float,
        high: float,
        low: float,
        direction: str,
    ) -> bool:
        """Return True if current candle is touching/entering the zone for the first time."""
        if direction == "bullish":
            # Candle low must dip into the zone (wick or open touches zone)
            return low <= zone.zone_high and (open_ >= zone.zone_low or close >= zone.zone_low)
        else:
            # Candle high must reach into the zone
            return high >= zone.zone_low and (open_ <= zone.zone_high or close <= zone.zone_high)

    def _has_rejection(
        self,
        zone: _Zone,
        open_: float,
        close: float,
        high: float,
        low: float,
        direction: str,
    ) -> bool:
        """Return True if the candle shows rejection from the zone.

        Two acceptance criteria:
          1. Wick > body in opposite direction (pin bar / hammer / shooting star)
          2. Engulfing in favour of the trend (close well past open)
        """
        body = abs(close - open_)

        if direction == "bullish":
            # Rejection upward: lower wick > body, or bullish engulf
            lower_wick = min(open_, close) - low
            bullish_body = close > open_
            return (lower_wick > body and low <= zone.zone_high) or (bullish_body and close > open_ + body * 0.5)
        else:
            # Rejection downward: upper wick > body, or bearish engulf
            upper_wick = high - max(open_, close)
            bearish_body = close < open_
            return (upper_wick > body and high >= zone.zone_low) or (bearish_body and close < open_ - body * 0.5)

    # ── Confluence check ───────────────────────────────────────────────────────

    def _has_confluence(
        self,
        zone: _Zone,
        ob_zones: list[_Zone],
        fvg_zones: list[_Zone],
    ) -> bool:
        """Return True if this zone overlaps with a zone of the other kind."""
        if zone.kind == "ob":
            others = fvg_zones
        else:
            others = ob_zones

        for other in others:
            # Overlap check: zones share any price range
            if zone.zone_low <= other.zone_high and zone.zone_high >= other.zone_low:
                return True
        return False

    # ── Utility ────────────────────────────────────────────────────────────────

    def _infer_symbol(self, candles: pd.DataFrame) -> str:
        if hasattr(candles, "attrs") and "symbol" in candles.attrs:
            return candles.attrs["symbol"]
        return "BTCUSDT"
