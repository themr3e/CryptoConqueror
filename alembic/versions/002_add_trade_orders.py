"""Add trade_orders table and binance testnet config columns.

Revision ID: 002
Revises: 001
Create Date: 2026-03-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: str = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trade_orders",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("signal_id", sa.BigInteger, sa.ForeignKey("signals.id"), nullable=False),
        sa.Column("broker_order_id", sa.String(50), nullable=True),
        sa.Column("order_role", sa.String(20), nullable=False),   # entry | stop_loss | take_profit_1
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(5), nullable=False),
        sa.Column("order_type", sa.String(30), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 8), nullable=False),
        sa.Column("price", sa.Numeric(18, 8), nullable=True),
        sa.Column("stop_price", sa.Numeric(18, 8), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="NEW"),
        sa.Column("raw_response", sa.Text, nullable=True),
        sa.Column("environment", sa.String(10), nullable=False, server_default="testnet"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index(
        "idx_trade_orders_signal",
        "trade_orders",
        ["signal_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_trade_orders_signal", table_name="trade_orders")
    op.drop_table("trade_orders")
