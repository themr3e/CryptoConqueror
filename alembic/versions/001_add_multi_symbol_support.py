"""Add multi-symbol support: widen price columns, add crypto fee config table.

This migration:
  1. Widens Candle price/volume columns to Numeric(18,8) for crypto precision
  2. Widens Candle.symbol to String(20)
  3. Adds Candle.source column ("twelve_data" | "binance_futures")
  4. Widens Signal price columns to Numeric(18,8)
  5. Widens Signal.symbol to String(20)
  6. Widens Outcome.exit_price to Numeric(18,8)
  7. Adds Outcome.pnl_usdt column for crypto P&L tracking
  8. Adds Strategy.asset_class column ("forex" | "crypto_futures")
  9. Adds Strategy.symbols column (JSON list of symbols)
 10. Creates crypto_fee_configs table with BTCUSDT/ETHUSDT seed data

Revision ID: 001
Revises:
Create Date: 2026-03-26
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    """Return True if *table_name* exists in the current schema."""
    result = op.get_bind().execute(
        sa.text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = :t"
        ),
        {"t": table_name},
    )
    return result.fetchone() is not None


def _column_exists(table_name: str, column_name: str) -> bool:
    """Return True if *column_name* exists in *table_name*."""
    result = op.get_bind().execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND table_name = :t AND column_name = :c"
        ),
        {"t": table_name, "c": column_name},
    )
    return result.fetchone() is not None


def upgrade() -> None:
    # -----------------------------------------------------------------------
    # 1. Candle table — widen symbol and price columns, add source
    # -----------------------------------------------------------------------
    if _table_exists("candles"):
        op.execute(
            "ALTER TABLE candles ALTER COLUMN symbol TYPE VARCHAR(20)"
        )
        op.execute(
            "ALTER TABLE candles ALTER COLUMN open TYPE NUMERIC(18,8) USING open::NUMERIC(18,8)"
        )
        op.execute(
            "ALTER TABLE candles ALTER COLUMN high TYPE NUMERIC(18,8) USING high::NUMERIC(18,8)"
        )
        op.execute(
            "ALTER TABLE candles ALTER COLUMN low TYPE NUMERIC(18,8) USING low::NUMERIC(18,8)"
        )
        op.execute(
            "ALTER TABLE candles ALTER COLUMN close TYPE NUMERIC(18,8) USING close::NUMERIC(18,8)"
        )
        op.execute(
            "ALTER TABLE candles ALTER COLUMN volume TYPE NUMERIC(24,8) USING volume::NUMERIC(24,8)"
        )
        if not _column_exists("candles", "source"):
            op.add_column(
                "candles",
                sa.Column(
                    "source",
                    sa.String(20),
                    nullable=False,
                    server_default="twelve_data",
                ),
            )

    # -----------------------------------------------------------------------
    # 2. Signal table — widen symbol and price columns
    # -----------------------------------------------------------------------
    if _table_exists("signals"):
        op.execute(
            "ALTER TABLE signals ALTER COLUMN symbol TYPE VARCHAR(20)"
        )
        op.execute(
            "ALTER TABLE signals ALTER COLUMN entry_price TYPE NUMERIC(18,8) "
            "USING entry_price::NUMERIC(18,8)"
        )
        op.execute(
            "ALTER TABLE signals ALTER COLUMN stop_loss TYPE NUMERIC(18,8) "
            "USING stop_loss::NUMERIC(18,8)"
        )
        op.execute(
            "ALTER TABLE signals ALTER COLUMN take_profit_1 TYPE NUMERIC(18,8) "
            "USING take_profit_1::NUMERIC(18,8)"
        )
        op.execute(
            "ALTER TABLE signals ALTER COLUMN take_profit_2 TYPE NUMERIC(18,8) "
            "USING take_profit_2::NUMERIC(18,8)"
        )

    # -----------------------------------------------------------------------
    # 3. Outcome table — widen exit_price, add pnl_usdt
    # -----------------------------------------------------------------------
    if _table_exists("outcomes"):
        op.execute(
            "ALTER TABLE outcomes ALTER COLUMN exit_price TYPE NUMERIC(18,8) "
            "USING exit_price::NUMERIC(18,8)"
        )
        if not _column_exists("outcomes", "pnl_usdt"):
            op.add_column(
                "outcomes",
                sa.Column("pnl_usdt", sa.Numeric(18, 8), nullable=True),
            )

    # -----------------------------------------------------------------------
    # 4. Strategy table — add asset_class and symbols
    # -----------------------------------------------------------------------
    if _table_exists("strategies"):
        if not _column_exists("strategies", "asset_class"):
            op.add_column(
                "strategies",
                sa.Column(
                    "asset_class",
                    sa.String(20),
                    nullable=False,
                    server_default="forex",
                ),
            )
        if not _column_exists("strategies", "symbols"):
            op.add_column(
                "strategies",
                sa.Column("symbols", sa.Text, nullable=True),
            )

    # -----------------------------------------------------------------------
    # 5. Create crypto_fee_configs table
    # -----------------------------------------------------------------------
    crypto_fee_table = op.create_table(
        "crypto_fee_configs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("taker_fee_rate", sa.Numeric(8, 6), nullable=False),
        sa.Column("maker_fee_rate", sa.Numeric(8, 6), nullable=False),
        sa.Column("funding_rate_8h", sa.Numeric(10, 8), nullable=True),
        sa.Column("tick_size", sa.Numeric(18, 8), nullable=False),
        sa.Column("lot_size", sa.Numeric(18, 8), nullable=False),
        sa.Column("leverage_max", sa.Integer, nullable=False, server_default="20"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("symbol", name="uq_crypto_fee_symbol"),
    )

    # Seed fee data for BTCUSDT and ETHUSDT
    op.bulk_insert(
        crypto_fee_table,
        [
            {
                "symbol": "BTCUSDT",
                "taker_fee_rate": "0.000450",
                "maker_fee_rate": "0.000200",
                "funding_rate_8h": "0.00010000",
                "tick_size": "0.10000000",
                "lot_size": "0.00100000",
                "leverage_max": 125,
                "created_at": datetime.now(timezone.utc),
                "updated_at": None,
            },
            {
                "symbol": "ETHUSDT",
                "taker_fee_rate": "0.000450",
                "maker_fee_rate": "0.000200",
                "funding_rate_8h": "0.00010000",
                "tick_size": "0.01000000",
                "lot_size": "0.00100000",
                "leverage_max": 100,
                "created_at": datetime.now(timezone.utc),
                "updated_at": None,
            },
        ],
    )


def downgrade() -> None:
    # -----------------------------------------------------------------------
    # Reverse in opposite order
    # -----------------------------------------------------------------------

    # 5. Drop crypto_fee_configs
    op.drop_table("crypto_fee_configs")

    # 4. Remove strategy columns
    op.drop_column("strategies", "symbols")
    op.drop_column("strategies", "asset_class")

    # 3. Remove pnl_usdt, restore exit_price
    op.drop_column("outcomes", "pnl_usdt")
    op.execute(
        "ALTER TABLE outcomes ALTER COLUMN exit_price TYPE NUMERIC(10,2) "
        "USING exit_price::NUMERIC(10,2)"
    )

    # 2. Restore signal columns
    op.execute(
        "ALTER TABLE signals ALTER COLUMN symbol TYPE VARCHAR(10)"
    )
    op.execute(
        "ALTER TABLE signals ALTER COLUMN entry_price TYPE NUMERIC(10,2) "
        "USING entry_price::NUMERIC(10,2)"
    )
    op.execute(
        "ALTER TABLE signals ALTER COLUMN stop_loss TYPE NUMERIC(10,2) "
        "USING stop_loss::NUMERIC(10,2)"
    )
    op.execute(
        "ALTER TABLE signals ALTER COLUMN take_profit_1 TYPE NUMERIC(10,2) "
        "USING take_profit_1::NUMERIC(10,2)"
    )
    op.execute(
        "ALTER TABLE signals ALTER COLUMN take_profit_2 TYPE NUMERIC(10,2) "
        "USING take_profit_2::NUMERIC(10,2)"
    )

    # 1. Restore candle columns
    op.drop_column("candles", "source")
    op.execute(
        "ALTER TABLE candles ALTER COLUMN symbol TYPE VARCHAR(10)"
    )
    op.execute(
        "ALTER TABLE candles ALTER COLUMN open TYPE NUMERIC(10,2) USING open::NUMERIC(10,2)"
    )
    op.execute(
        "ALTER TABLE candles ALTER COLUMN high TYPE NUMERIC(10,2) USING high::NUMERIC(10,2)"
    )
    op.execute(
        "ALTER TABLE candles ALTER COLUMN low TYPE NUMERIC(10,2) USING low::NUMERIC(10,2)"
    )
    op.execute(
        "ALTER TABLE candles ALTER COLUMN close TYPE NUMERIC(10,2) USING close::NUMERIC(10,2)"
    )
    op.execute(
        "ALTER TABLE candles ALTER COLUMN volume TYPE NUMERIC(15,2) USING volume::NUMERIC(15,2)"
    )
