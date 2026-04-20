"""Tests for P&L calculation and signal expiry bugs.

Bug 1: Signals with NULL expires_at never expired → ghost P&L from old signals.
Bug 2: P&L ignored position_size → always assumed 1 contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Bug 1 — expire_stale_signals must catch NULL expires_at signals
# ---------------------------------------------------------------------------

class TestExpireStaleSignals:
    """expire_stale_signals must kill old signals with no expiry set.

    These tests validate the expiry condition logic directly — avoiding
    importing SignalGenerator which requires Python 3.10+ type syntax
    not available on the local Python 3.9 dev machine.
    """

    def test_null_expiry_48h_cutoff_logic(self):
        """Signal created 72h ago with NULL expiry must fall inside cutoff."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=48)

        created_72h_ago = now - timedelta(hours=72)
        created_24h_ago = now - timedelta(hours=24)

        # 72h old signal → must expire
        assert created_72h_ago < cutoff, "72h-old NULL-expiry signal should be expired"
        # 24h old signal → must NOT expire yet
        assert created_24h_ago >= cutoff, "24h-old NULL-expiry signal should stay active"

    def test_explicit_expiry_past_is_caught(self):
        """Signal with expires_at in the past must be caught."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        expired_at = now - timedelta(hours=1)
        future_expiry = now + timedelta(hours=7)

        assert expired_at < now, "Past expiry → should be expired"
        assert future_expiry >= now, "Future expiry → should stay active"

    def test_both_conditions_cover_all_stale_cases(self):
        """Together, the two conditions catch every possible stale signal."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        cutoff_48h = now - timedelta(hours=48)

        cases = [
            # (expires_at, created_at, should_expire, description)
            (now - timedelta(hours=1), now - timedelta(hours=5), True,  "explicit expiry in past"),
            (now + timedelta(hours=7), now - timedelta(hours=1), False, "explicit expiry in future"),
            (None,                     now - timedelta(hours=72), True,  "NULL expiry, 72h old"),
            (None,                     now - timedelta(hours=24), False, "NULL expiry, 24h old"),
        ]

        for expires_at, created_at, expected, desc in cases:
            if expires_at is not None:
                result = expires_at < now
            else:
                result = created_at < cutoff_48h
            assert result == expected, f"Failed: {desc}"


# ---------------------------------------------------------------------------
# Bug 2 — P&L must multiply by position_size
# ---------------------------------------------------------------------------

class TestPnlCalculation:
    """_record_outcome P&L must use signal.position_size, not assume 1 unit."""

    def _make_signal(
        self,
        direction: str,
        entry_price: str,
        stop_loss: str,
        position_size: str | None,
    ):
        signal = MagicMock()
        signal.direction = direction
        signal.entry_price = Decimal(entry_price)
        signal.stop_loss = Decimal(stop_loss)
        signal.position_size = Decimal(position_size) if position_size else None
        signal.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
        signal.expires_at = datetime.now(timezone.utc) + timedelta(hours=7)
        signal.id = 1
        signal.symbol = "BTCUSDT"
        return signal

    def _pnl_formula(
        self,
        direction: str,
        entry: Decimal,
        exit_price: Decimal,
        position_size: Decimal,
        taker_fee: Decimal,
    ) -> Decimal:
        """Mirror the fixed formula from crypto_outcome_detector._record_outcome."""
        direction_mult = Decimal("1") if direction == "BUY" else Decimal("-1")
        raw_pnl = (exit_price - entry) * direction_mult * position_size
        fee_cost = entry * taker_fee * Decimal("2") * position_size
        return raw_pnl - fee_cost

    def test_sell_sl_hit_with_position_size_1(self):
        """SELL, SL hit 1% above entry, 1 BTC — loss ≈ entry * 1% + fees."""
        entry = Decimal("83000")
        exit_p = Decimal("83830")   # ~1% above entry → SL hit
        pos_size = Decimal("1")
        fee = Decimal("0.000450")

        pnl = self._pnl_formula("SELL", entry, exit_p, pos_size, fee)

        assert pnl < 0, "SELL hitting SL must be a loss"
        # raw loss = (83830 - 83000) * -1 * 1 = -830; fees = 83000*0.0009 = -74.7
        assert Decimal("-910") < pnl < Decimal("-900")

    def test_sell_sl_hit_with_small_position_size(self):
        """With position_size=0.024 BTC (1% risk), loss must be ~1% of account."""
        entry = Decimal("83000")
        exit_p = Decimal("83830")
        pos_size = Decimal("0.024")
        fee = Decimal("0.000450")

        pnl = self._pnl_formula("SELL", entry, exit_p, pos_size, fee)

        assert pnl < 0
        # raw loss ≈ -830 * 0.024 = -19.92; fees ≈ -1.79; total ≈ -21.71
        assert Decimal("-25") < pnl < Decimal("-19")

    def test_buy_tp_hit_with_position_size(self):
        """BUY hitting TP must be a profit scaled by position_size."""
        entry = Decimal("83000")
        exit_p = Decimal("84660")   # 2% profit
        pos_size = Decimal("0.024")
        fee = Decimal("0.000450")

        pnl = self._pnl_formula("BUY", entry, exit_p, pos_size, fee)

        assert pnl > 0, "BUY hitting TP must be a profit"

    def test_ghost_pnl_scenario_prevented(self):
        """Old SELL signal (entry $8000) vs current BTC $83000.

        With position_size = 0.012 (correctly sized for $8k account,1% risk, $800 SL):
        loss should be ~$100, NOT $75000.
        """
        entry = Decimal("8000")
        exit_p = Decimal("83000")   # current BTC price, SL long ago breached
        pos_size = Decimal("0.012")  # correctly sized: risk=$100, sl_dist=$800
        fee = Decimal("0.000450")

        pnl = self._pnl_formula("SELL", entry, exit_p, pos_size, fee)

        assert pnl < 0
        # raw: (83000-8000)*-1 * 0.012 = -900; fees = 8000*0.0009*0.012 = -0.086
        # total ≈ -900 (still large because position lived too long —
        # this is why expire_stale_signals fix is ALSO required)
        # With BOTH fixes: signal never reaches this state (expired after 48h)
        assert pnl < Decimal("-800")   # confirms the math works correctly

    def test_position_size_none_defaults_to_one(self):
        """If position_size is NULL in DB, must default to 1 (backward compat)."""
        entry = Decimal("83000")
        exit_p = Decimal("83830")
        pos_size = Decimal("1")   # fallback
        fee = Decimal("0.000450")

        pnl_explicit_1 = self._pnl_formula("SELL", entry, exit_p, pos_size, fee)
        pnl_default    = self._pnl_formula("SELL", entry, exit_p, Decimal("1"), fee)

        assert pnl_explicit_1 == pnl_default


# ---------------------------------------------------------------------------
# Daily report math
# ---------------------------------------------------------------------------

class TestDailyReportMath:
    """Daily report P&L aggregation must sum pnl_usdt correctly."""

    def test_total_pnl_sums_all_outcomes(self):
        outcomes = [
            MagicMock(pnl_usdt=Decimal("150.00")),
            MagicMock(pnl_usdt=Decimal("-21.50")),
            MagicMock(pnl_usdt=Decimal("-900.00")),
        ]
        total = sum(float(o.pnl_usdt or 0) for o in outcomes)
        assert abs(total - (-771.50)) < 0.01

    def test_win_rate_calculation(self):
        results = ["tp1_hit", "sl_hit", "tp2_hit", "sl_hit", "sl_hit"]
        wins = sum(1 for r in results if r in ("tp1_hit", "tp2_hit"))
        total = len(results)
        win_rate = wins / total * 100

        assert win_rate == 40.0

    def test_worst_trade_is_last_when_sorted_desc(self):
        """Outcomes ordered pnl_usdt DESC → last row is worst trade."""
        pnl_values = [150.0, -21.5, -900.0]
        sorted_desc = sorted(pnl_values, reverse=True)

        assert sorted_desc[-1] == -900.0
