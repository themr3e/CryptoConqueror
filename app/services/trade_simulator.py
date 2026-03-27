"""Trade simulator for backtesting signal outcomes against OHLC candle data."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

import pandas as pd

from app.strategies.base import CandidateSignal


class TradeOutcome(str, Enum):
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    SL_HIT = "sl_hit"
    EXPIRED = "expired"


@dataclass
class SimulatedTrade:
    """Result of simulating a single trade."""
    signal: CandidateSignal
    outcome: TradeOutcome
    exit_price: Decimal
    pnl_pips: Decimal
    bars_held: int
    spread_cost: Decimal


class TradeSimulator:
    """Simulates trade outcomes by walking signals through OHLC price data."""

    MAX_BARS_FORWARD = 72  # Maximum bars to hold a trade before expiry
    PIP_VALUE = 1.0        # Crypto: raw USDT price difference (1 pip = $1)

    def simulate_trade(
        self,
        signal: CandidateSignal,
        candles: pd.DataFrame,
        entry_bar_idx: int,
        spread: Decimal,
    ) -> SimulatedTrade:
        """Simulate a single trade from entry bar forward.

        Args:
            signal: CandidateSignal with entry, SL, and TP prices.
            candles: Full OHLC DataFrame (chronological order).
            entry_bar_idx: Index of the bar where signal was generated.
            spread: Spread in price units.

        Returns:
            SimulatedTrade with outcome, exit price, PnL, and duration.
        """
        is_buy = signal.direction.value == "BUY"
        entry = float(signal.entry_price)
        sl = float(signal.stop_loss)
        tp1 = float(signal.take_profit_1)
        tp2 = float(signal.take_profit_2) if signal.take_profit_2 is not None else None
        spread_f = float(spread)

        # Adjust entry for spread (ask price for buys)
        actual_entry = entry + spread_f if is_buy else entry

        outcome = TradeOutcome.EXPIRED
        exit_price = actual_entry
        bars_held = 0

        for i in range(entry_bar_idx + 1, min(entry_bar_idx + self.MAX_BARS_FORWARD + 1, len(candles))):
            bar = candles.iloc[i]
            high = float(bar["high"])
            low = float(bar["low"])
            bars_held = i - entry_bar_idx

            if is_buy:
                # BUY: SL if low drops to/below SL, TP if high reaches TP
                sl_hit = low <= sl
                tp2_hit = tp2 is not None and high >= tp2
                tp1_hit = high >= tp1
            else:
                # SELL: SL if high (+ spread as ask) reaches SL, TP if low drops to TP
                ask_high = high + spread_f
                sl_hit = ask_high >= sl
                tp2_hit = tp2 is not None and low <= tp2
                tp1_hit = low <= tp1

            # SL takes priority over TP
            if sl_hit:
                outcome = TradeOutcome.SL_HIT
                exit_price = sl
                break
            elif tp2_hit:
                outcome = TradeOutcome.TP2_HIT
                exit_price = tp2
                break
            elif tp1_hit:
                outcome = TradeOutcome.TP1_HIT
                exit_price = tp1
                break

        # Compute PnL in pips
        if is_buy:
            pnl = (exit_price - actual_entry) / self.PIP_VALUE
        else:
            pnl = (actual_entry - exit_price) / self.PIP_VALUE

        return SimulatedTrade(
            signal=signal,
            outcome=outcome,
            exit_price=Decimal(str(round(exit_price, 2))),
            pnl_pips=Decimal(str(round(pnl, 2))),
            bars_held=bars_held,
            spread_cost=spread,
        )

    def simulate_signals(
        self,
        signals: list[CandidateSignal],
        candles: pd.DataFrame,
        spread_model,
    ) -> list[SimulatedTrade]:
        """Simulate a batch of signals against candle data."""
        results = []
        for signal in signals:
            # Find entry bar index (last bar at or before signal timestamp)
            ts = signal.timestamp
            matching = candles[candles["timestamp"] <= ts]
            if matching.empty:
                continue
            entry_idx = matching.index[-1]
            spread = spread_model.get_spread(ts)
            trade = self.simulate_trade(signal, candles, entry_idx, spread)
            results.append(trade)
        return results
