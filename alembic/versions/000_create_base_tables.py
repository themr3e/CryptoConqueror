"""Create all base tables (initial schema).

This is the initial migration that creates the base table structure
before subsequent migrations alter/extend it.

Revision ID: 000
Revises:
Create Date: 2026-03-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "000"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -----------------------------------------------------------------------
    # 1. strategies
    # -----------------------------------------------------------------------
    op.create_table(
        "strategies",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(50), nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 2. candles (original narrow types — migration 001 widens them)
    # -----------------------------------------------------------------------
    op.create_table(
        "candles",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("timeframe", sa.String(5), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(10, 2), nullable=False),
        sa.Column("high", sa.Numeric(10, 2), nullable=False),
        sa.Column("low", sa.Numeric(10, 2), nullable=False),
        sa.Column("close", sa.Numeric(10, 2), nullable=False),
        sa.Column("volume", sa.Numeric(15, 2), nullable=True),
        sa.UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle_identity"),
    )
    op.create_index("idx_candles_lookup", "candles", ["symbol", "timeframe", "timestamp"])

    # -----------------------------------------------------------------------
    # 3. signals
    # -----------------------------------------------------------------------
    op.create_table(
        "signals",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("strategy_id", sa.BigInteger, sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("timeframe", sa.String(5), nullable=False),
        sa.Column("direction", sa.String(5), nullable=False),
        sa.Column("entry_price", sa.Numeric(10, 2), nullable=False),
        sa.Column("stop_loss", sa.Numeric(10, 2), nullable=False),
        sa.Column("take_profit_1", sa.Numeric(10, 2), nullable=False),
        sa.Column("take_profit_2", sa.Numeric(10, 2), nullable=False),
        sa.Column("risk_reward", sa.Numeric(5, 2), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 2), nullable=False),
        sa.Column("reasoning", sa.Text, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )

    # -----------------------------------------------------------------------
    # 4. outcomes
    # -----------------------------------------------------------------------
    op.create_table(
        "outcomes",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "signal_id",
            sa.BigInteger,
            sa.ForeignKey("signals.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("result", sa.String(20), nullable=False),
        sa.Column("exit_price", sa.Numeric(10, 2), nullable=False),
        sa.Column("pnl_pips", sa.Numeric(10, 2), nullable=False),
        sa.Column("duration_minutes", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )

    # -----------------------------------------------------------------------
    # 5. strategy_performance
    # -----------------------------------------------------------------------
    op.create_table(
        "strategy_performance",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("strategy_id", sa.BigInteger, sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("period", sa.String(10), nullable=False),
        sa.Column("win_rate", sa.Numeric(10, 4), nullable=False),
        sa.Column("profit_factor", sa.Numeric(10, 4), nullable=False),
        sa.Column("avg_rr", sa.Numeric(10, 4), nullable=False),
        sa.Column("total_signals", sa.Integer, nullable=False),
        sa.Column("is_degraded", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column(
            "calculated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )

    # -----------------------------------------------------------------------
    # 6. backtest_results
    # -----------------------------------------------------------------------
    op.create_table(
        "backtest_results",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("strategy_id", sa.BigInteger, sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("timeframe", sa.String(5), nullable=False),
        sa.Column("window_days", sa.Integer, nullable=False),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("win_rate", sa.Numeric(10, 4), nullable=True),
        sa.Column("profit_factor", sa.Numeric(10, 4), nullable=True),
        sa.Column("sharpe_ratio", sa.Numeric(10, 4), nullable=True),
        sa.Column("max_drawdown", sa.Numeric(10, 4), nullable=True),
        sa.Column("expectancy", sa.Numeric(10, 4), nullable=True),
        sa.Column("total_trades", sa.Integer, nullable=False),
        sa.Column("is_walk_forward", sa.Boolean, nullable=True),
        sa.Column("is_overfitted", sa.Boolean, nullable=True),
        sa.Column("walk_forward_efficiency", sa.Numeric(10, 4), nullable=True),
        sa.Column("spread_model", sa.String(20), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )

    # -----------------------------------------------------------------------
    # 7. optimized_params
    # -----------------------------------------------------------------------
    op.create_table(
        "optimized_params",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("strategy_id", sa.BigInteger, sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("strategy_name", sa.String(50), nullable=False),
        sa.Column("params", sa.JSON, nullable=False),
        sa.Column("win_rate", sa.Numeric(10, 4), nullable=True),
        sa.Column("profit_factor", sa.Numeric(10, 4), nullable=True),
        sa.Column("sharpe_ratio", sa.Numeric(10, 4), nullable=True),
        sa.Column("expectancy", sa.Numeric(10, 4), nullable=True),
        sa.Column("total_trades", sa.Integer, nullable=False),
        sa.Column("wfe_ratio", sa.Numeric(10, 4), nullable=True),
        sa.Column("is_overfitted", sa.Boolean, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("combinations_tested", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("optimized_params")
    op.drop_table("backtest_results")
    op.drop_table("strategy_performance")
    op.drop_table("outcomes")
    op.drop_table("signals")
    op.drop_index("idx_candles_lookup", table_name="candles")
    op.drop_table("candles")
    op.drop_table("strategies")
