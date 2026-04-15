"""Footprint / order-flow imbalance strategy.

Generates signals from pre-computed :class:`FootprintBar` rows rather than
from raw OHLCV candles. The signal_generator attaches the most recent
footprint bars to ``candles.attrs["footprint_bars"]`` as a list of dicts
with the same column names as :class:`FootprintBar`.

Entry logic (mirrors TradingIQ's Pine Script "Stacked Imbalance" alert):

    - BUY  when the most recent bar has ``stacked_imb_count ≥ min_stack``
      on the ``"BUY"`` side **and** the rolling ``cum_delta`` is rising.
    - SELL when the same conditions hold on the ``"SELL"`` side with a
      falling ``cum_delta``.

Risk levels come from the profile itself:
    - Long  : SL = VAL, TP1 = POC + 1 × risk, TP2 = POC + 2 × risk.
    - Short : SL = VAH, TP1 = POC − 1 × risk, TP2 = POC − 2 × risk.

When footprint bars are missing (older deployment without the rollup
job running yet), the strategy emits zero signals and logs a warning.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from app.strategies.base import BaseStrategy, CandidateSignal, SignalDirection


class FootprintImbalanceStrategy(BaseStrategy):
    """Stacked-imbalance strategy fed by rolled-up footprint bars."""

    NAME = "footprint_imbalance"
    ASSET_CLASS = "crypto_futures"
    DEFAULT_PARAMS: dict[str, float] = {
        "MIN_STACK": 3,             # consecutive imbalance levels required
        "MIN_CUM_DELTA_STREAK": 3,  # bars the cum_delta must confirm direction
        "TP1_RR": 1.5,
        "TP2_RR": 3.0,
        "MIN_CONFIDENCE": 55.0,
    }

    def generate_signals(self, candles: pd.DataFrame) -> list[CandidateSignal]:
        """Consume footprint bars from ``candles.attrs["footprint_bars"]``."""
        bars: list[dict[str, Any]] | None = (
            candles.attrs.get("footprint_bars") if hasattr(candles, "attrs") else None
        )
        if not bars:
            logger.debug(
                "FootprintImbalance: no footprint_bars attached for {} — skipping.",
                candles.attrs.get("symbol", "?") if hasattr(candles, "attrs") else "?",
            )
            return []

        # `bars` is assumed newest-last
        if len(bars) < int(self.params["MIN_CUM_DELTA_STREAK"]) + 1:
            return []

        min_stack = int(self.params["MIN_STACK"])
        streak_len = int(self.params["MIN_CUM_DELTA_STREAK"])
        tp1_rr = float(self.params["TP1_RR"])
        tp2_rr = float(self.params["TP2_RR"])
        min_conf = float(self.params["MIN_CONFIDENCE"])

        last = bars[-1]
        count = int(last.get("stacked_imb_count") or 0)
        side = last.get("stacked_imb_side")

        if count < min_stack or side not in ("BUY", "SELL"):
            return []

        # Cumulative-delta direction must agree with the imbalance side.
        streak = bars[-streak_len:]
        cds = [float(b["cum_delta"]) for b in streak]
        cd_rising = all(cds[i] <= cds[i + 1] for i in range(len(cds) - 1))
        cd_falling = all(cds[i] >= cds[i + 1] for i in range(len(cds) - 1))

        if side == "BUY" and not cd_rising:
            return []
        if side == "SELL" and not cd_falling:
            return []

        entry = float(last["close"])
        poc = float(last["poc"])
        vah = float(last["vah"])
        val = float(last["val"])

        if side == "BUY":
            sl = val if val < entry else entry * 0.985
            risk = entry - sl
            if risk <= 0:
                return []
            tp1 = max(poc, entry + tp1_rr * risk)
            tp2 = entry + tp2_rr * risk
            direction = SignalDirection.BUY
            reasoning = (
                f"Footprint BUY: {count} stacked BUY imbalances, "
                f"cum_delta rising ({cds[0]:.0f}→{cds[-1]:.0f}); "
                f"POC={poc:.4f} VAL={val:.4f}"
            )
        else:
            sl = vah if vah > entry else entry * 1.015
            risk = sl - entry
            if risk <= 0:
                return []
            tp1 = min(poc, entry - tp1_rr * risk)
            tp2 = entry - tp2_rr * risk
            direction = SignalDirection.SELL
            reasoning = (
                f"Footprint SELL: {count} stacked SELL imbalances, "
                f"cum_delta falling ({cds[0]:.0f}→{cds[-1]:.0f}); "
                f"POC={poc:.4f} VAH={vah:.4f}"
            )

        confidence = min(min_conf + 5.0 * (count - min_stack), 90.0)

        symbol = (
            candles.attrs.get("symbol", "UNKNOWN")
            if hasattr(candles, "attrs")
            else "UNKNOWN"
        )

        return [CandidateSignal(
            strategy_name=self.NAME,
            symbol=symbol,
            timeframe="M1",
            direction=direction,
            entry_price=self._to_decimal(entry),
            stop_loss=self._to_decimal(sl),
            take_profit_1=self._to_decimal(tp1),
            take_profit_2=self._to_decimal(tp2),
            risk_reward=self._to_decimal(tp1_rr, 2),
            confidence=self._to_decimal(confidence, 2),
            reasoning=reasoning,
            session="24h",
        )]
