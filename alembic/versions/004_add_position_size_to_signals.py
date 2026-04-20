"""Add position_size column to signals table.

Revision ID: 004
Revises: 003
Create Date: 2026-04-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "signals",
        sa.Column("position_size", sa.Numeric(18, 8), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("signals", "position_size")
