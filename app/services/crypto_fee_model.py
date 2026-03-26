"""Crypto futures fee model.

Replaces the role of ``SessionSpreadModel`` for crypto assets.
Crypto perpetual futures have:
  - Taker fee (paid on market-order fills)
  - Funding rate (paid/received every 8h for holding positions)

No session-based spread applies — crypto markets run 24/7.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.crypto_fee_config import CryptoFeeConfig

# Default Binance Futures standard-tier fees
_DEFAULT_TAKER_FEE = Decimal("0.000450")   # 0.045%
_DEFAULT_MAKER_FEE = Decimal("0.000200")   # 0.020%


class CryptoFeeModel:
    """Calculates trading costs for crypto futures positions."""

    async def get_taker_fee(
        self,
        session: AsyncSession,
        symbol: str,
    ) -> Decimal:
        """Load taker fee for the given symbol from DB; fall back to default.

        Args:
            session: Async DB session.
            symbol:  e.g. ``"BTCUSDT"``.

        Returns:
            Taker fee rate as Decimal (e.g. ``Decimal("0.000450")``).
        """
        stmt = select(CryptoFeeConfig.taker_fee_rate).where(
            CryptoFeeConfig.symbol == symbol
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        return Decimal(str(row)) if row is not None else _DEFAULT_TAKER_FEE

    def calculate_round_trip_cost(
        self,
        entry_price: Decimal,
        position_size: Decimal,
        taker_fee: Decimal,
    ) -> Decimal:
        """Return total USDT cost for one round-trip trade (entry + exit fills).

        Args:
            entry_price:   Fill price for entry.
            position_size: Number of contracts / base currency units.
            taker_fee:     Taker fee rate (e.g. ``Decimal("0.000450")``).

        Returns:
            Total fee cost in USDT.
        """
        notional = entry_price * position_size
        return notional * taker_fee * 2  # entry + exit

    def adjust_take_profit_for_fees(
        self,
        direction: str,
        entry_price: Decimal,
        take_profit: Decimal,
        taker_fee: Decimal,
    ) -> Decimal:
        """Nudge take-profit away from entry to ensure net-positive after fees.

        The fee adjustment is ``entry_price * taker_fee * 2`` expressed as a
        price distance from entry.  This guarantees the trade is net-profitable
        after round-trip costs even when TP is the exit level.

        Args:
            direction:   ``"BUY"`` or ``"SELL"``.
            entry_price: Entry fill price.
            take_profit: Original TP level.
            taker_fee:   Taker fee rate.

        Returns:
            Fee-adjusted TP price.
        """
        fee_distance = entry_price * taker_fee * 2
        if direction == "BUY":
            return take_profit + fee_distance
        return take_profit - fee_distance

    def spread_cost_pct(self, taker_fee: Decimal) -> Decimal:
        """Return round-trip cost as a percentage of notional.

        Equivalent to ``SessionSpreadModel.get_spread()`` but for crypto.

        Returns:
            e.g. ``Decimal("0.00090")`` for 0.09% round-trip.
        """
        return taker_fee * 2

    def estimate_position_size(
        self,
        account_balance: Decimal,
        risk_pct: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
    ) -> Decimal:
        """Calculate position size in contracts given account risk parameters.

        Args:
            account_balance: Total account equity in USDT.
            risk_pct:        Fraction of account to risk, e.g. ``Decimal("0.01")``.
            entry_price:     Entry fill price.
            stop_loss:       Stop loss price.

        Returns:
            Position size (base currency units), rounded to 3 decimal places.
        """
        risk_amount = account_balance * risk_pct
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return Decimal("0")
        size = risk_amount / price_risk
        return round(size, 3)
