"""Add raw_trades and footprint_bars tables for order-flow engine.

Revision ID: 004
Revises: 003
Create Date: 2026-04-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "raw_trades",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("trade_id", sa.BigInteger, nullable=False),
        sa.Column("price", sa.Numeric(18, 8), nullable=False),
        sa.Column("qty", sa.Numeric(24, 8), nullable=False),
        sa.Column("is_buyer_maker", sa.Boolean, nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(20), nullable=False, server_default="agg_trades"),
        sa.UniqueConstraint("symbol", "trade_id", name="uq_raw_trade_identity"),
    )
    op.create_index("idx_raw_trades_symbol_ts", "raw_trades", ["symbol", "ts"])

    op.create_table(
        "footprint_bars",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tick_size", sa.Numeric(18, 8), nullable=False),
        sa.Column("levels", postgresql.JSONB, nullable=False),
        sa.Column("poc", sa.Numeric(18, 8), nullable=False),
        sa.Column("vah", sa.Numeric(18, 8), nullable=False),
        sa.Column("val", sa.Numeric(18, 8), nullable=False),
        sa.Column("total_vol", sa.Numeric(24, 8), nullable=False),
        sa.Column("buy_vol", sa.Numeric(24, 8), nullable=False),
        sa.Column("sell_vol", sa.Numeric(24, 8), nullable=False),
        sa.Column("delta", sa.Numeric(24, 8), nullable=False),
        sa.Column("cum_delta", sa.Numeric(24, 8), nullable=False),
        sa.Column("stacked_imb_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("stacked_imb_side", sa.String(5), nullable=True),
        sa.Column("high", sa.Numeric(18, 8), nullable=False),
        sa.Column("low", sa.Numeric(18, 8), nullable=False),
        sa.Column("close", sa.Numeric(18, 8), nullable=False),
        sa.Column("source", sa.String(20), nullable=False, server_default="agg_trades"),
        sa.UniqueConstraint("symbol", "ts", name="uq_footprint_bar_identity"),
    )
    op.create_index("idx_footprint_bars_symbol_ts", "footprint_bars", ["symbol", "ts"])


def downgrade() -> None:
    op.drop_index("idx_footprint_bars_symbol_ts", table_name="footprint_bars")
    op.drop_table("footprint_bars")
    op.drop_index("idx_raw_trades_symbol_ts", table_name="raw_trades")
    op.drop_table("raw_trades")
