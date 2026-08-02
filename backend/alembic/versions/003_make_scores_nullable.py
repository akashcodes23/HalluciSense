"""Make confidence_gap and consistency_failure nullable in verification_reports and sentence_analyses.

Revision ID: 003_make_scores_nullable
Revises: 002_add_corrected_response
Create Date: 2026-07-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003_make_scores_nullable'
down_revision: Union[str, None] = '002_add_corrected_response'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'verification_reports',
        'confidence_gap_score',
        existing_type=sa.Float(),
        nullable=True,
    )
    op.alter_column(
        'verification_reports',
        'consistency_failure_score',
        existing_type=sa.Float(),
        nullable=True,
    )
    op.alter_column(
        'sentence_analyses',
        'confidence_gap',
        existing_type=sa.Float(),
        nullable=True,
    )
    op.alter_column(
        'sentence_analyses',
        'consistency_failure',
        existing_type=sa.Float(),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        'verification_reports',
        'confidence_gap_score',
        existing_type=sa.Float(),
        nullable=False,
    )
    op.alter_column(
        'verification_reports',
        'consistency_failure_score',
        existing_type=sa.Float(),
        nullable=False,
    )
    op.alter_column(
        'sentence_analyses',
        'confidence_gap',
        existing_type=sa.Float(),
        nullable=False,
    )
    op.alter_column(
        'sentence_analyses',
        'consistency_failure',
        existing_type=sa.Float(),
        nullable=False,
    )
