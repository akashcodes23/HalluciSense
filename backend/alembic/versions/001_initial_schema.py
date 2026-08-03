"""Initial Schema Migration for HalluciSense SaaS.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-08-03
"""

from alembic import op
import sqlalchemy as sa


def upgrade():
    # Upgrade logic for normalized tables
    op.create_table(
        "organizations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=100), unique=True, nullable=False),
        sa.Column("plan_tier", sa.String(length=50), default="ENTERPRISE"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("organizations")
