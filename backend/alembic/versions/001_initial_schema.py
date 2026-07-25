"""Initial schema migration: users, chats, messages, verification_reports, sentence_analyses, evidence_items, analytics_events.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-07-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(length=320), nullable=False),
        sa.Column('hashed_password', sa.Text(), nullable=True),
        sa.Column('full_name', sa.String(length=255), nullable=False, server_default=""),
        sa.Column('avatar_url', sa.Text(), nullable=True),
        sa.Column('role', sa.String(length=20), nullable=False, server_default="USER"),
        sa.Column('preferred_model', sa.String(length=100), nullable=False, server_default="gemini-2.0-flash"),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default="true"),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default="false"),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_role'), 'users', ['role'], unique=False)

    # 2. Chats table
    op.create_table(
        'chats',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(length=512), nullable=False, server_default="New Chat"),
        sa.Column('model_used', sa.String(length=100), nullable=False, server_default="gemini-2.0-flash"),
        sa.Column('is_archived', sa.Boolean(), nullable=False, server_default="false"),
        sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_chats_id'), 'chats', ['id'], unique=False)
    op.create_index(op.f('ix_chats_user_id'), 'chats', ['user_id'], unique=False)

    # 3. Messages table
    op.create_table(
        'messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('chat_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('chats.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('role', sa.String(length=20), nullable=False, server_default="USER"),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('raw_logits', sa.JSON(), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('verification_status', sa.String(length=20), nullable=False, server_default="PENDING"),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_messages_id'), 'messages', ['id'], unique=False)
    op.create_index(op.f('ix_messages_chat_id'), 'messages', ['chat_id'], unique=False)
    op.create_index(op.f('ix_messages_user_id'), 'messages', ['user_id'], unique=False)
    op.create_index(op.f('ix_messages_verification_status'), 'messages', ['verification_status'], unique=False)

    # 4. Verification Reports table
    op.create_table(
        'verification_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('message_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('messages.id', ondelete='CASCADE'), nullable=False),
        sa.Column('overall_h_score', sa.Float(), nullable=False),
        sa.Column('overall_risk_level', sa.String(length=30), nullable=False, server_default="NEEDS_VERIFICATION"),
        sa.Column('factual_error_score', sa.Float(), nullable=False, server_default="0.0"),
        sa.Column('confidence_gap_score', sa.Float(), nullable=False, server_default="0.0"),
        sa.Column('consistency_failure_score', sa.Float(), nullable=False, server_default="0.0"),
        sa.Column('weights_used', sa.JSON(), nullable=False),
        sa.Column('pillar1_summary', sa.JSON(), nullable=True),
        sa.Column('pillar2_summary', sa.JSON(), nullable=True),
        sa.Column('pillar3_summary', sa.JSON(), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_verification_reports_id'), 'verification_reports', ['id'], unique=False)
    op.create_index(op.f('ix_verification_reports_message_id'), 'verification_reports', ['message_id'], unique=True)
    op.create_index(op.f('ix_verification_reports_overall_risk_level'), 'verification_reports', ['overall_risk_level'], unique=False)

    # 5. Sentence Analyses table
    op.create_table(
        'sentence_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('report_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('verification_reports.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sentence_index', sa.Integer(), nullable=False),
        sa.Column('sentence_text', sa.Text(), nullable=False),
        sa.Column('start_char', sa.Integer(), nullable=False, server_default="0"),
        sa.Column('end_char', sa.Integer(), nullable=False, server_default="0"),
        sa.Column('h_score', sa.Float(), nullable=False),
        sa.Column('risk_level', sa.String(length=30), nullable=False, server_default="NEEDS_VERIFICATION"),
        sa.Column('color_code', sa.String(length=10), nullable=False, server_default="#F59E0B"),
        sa.Column('factual_error', sa.Float(), nullable=False, server_default="0.0"),
        sa.Column('confidence_gap', sa.Float(), nullable=False, server_default="0.0"),
        sa.Column('consistency_failure', sa.Float(), nullable=False, server_default="0.0"),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_sentence_analyses_id'), 'sentence_analyses', ['id'], unique=False)
    op.create_index(op.f('ix_sentence_analyses_report_id'), 'sentence_analyses', ['report_id'], unique=False)

    # 6. Evidence Items table
    op.create_table(
        'evidence_items',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('sentence_analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sentence_analyses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('claim', sa.Text(), nullable=False),
        sa.Column('snippet', sa.Text(), nullable=False),
        sa.Column('source_name', sa.String(length=512), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('similarity_score', sa.Float(), nullable=False, server_default="0.0"),
        sa.Column('is_supporting', sa.Boolean(), nullable=False, server_default="true"),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_evidence_items_id'), 'evidence_items', ['id'], unique=False)
    op.create_index(op.f('ix_evidence_items_sentence_analysis_id'), 'evidence_items', ['sentence_analysis_id'], unique=False)

    # 7. Analytics Events table
    op.create_table(
        'analytics_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_analytics_events_id'), 'analytics_events', ['id'], unique=False)
    op.create_index(op.f('ix_analytics_events_event_type'), 'analytics_events', ['event_type'], unique=False)
    op.create_index(op.f('ix_analytics_events_user_id'), 'analytics_events', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_table('analytics_events')
    op.drop_table('evidence_items')
    op.drop_table('sentence_analyses')
    op.drop_table('verification_reports')
    op.drop_table('messages')
    op.drop_table('chats')
    op.drop_table('users')
