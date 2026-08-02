"""Add corrected_response column to verification_reports.

Revision ID: 002_add_corrected_response
Revises: 001_initial_schema
Create Date: 2026-07-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002_add_corrected_response'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'verification_reports',
        sa.Column('corrected_response', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('verification_reports', 'corrected_response')
