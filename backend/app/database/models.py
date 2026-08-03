"""
HalluciSense SaaS — Module 12.2: PostgreSQL Normalized Database Schemas
========================================================================
SQLAlchemy 2.0 ORM models for production enterprise database persistence.
Models:
  - Organization
  - User
  - Project
  - APIKey
  - VerificationSession
  - EvidenceCache
  - ProviderResponse
  - AuditLog
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""
    pass


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    plan_tier: Mapped[str] = mapped_column(String(50), default="ENTERPRISE")
    monthly_quota_verifications: Mapped[int] = mapped_column(Integer, default=100000)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    users: Mapped[List[User]] = relationship("User", back_populates="organization")
    projects: Mapped[List[Project]] = relationship("Project", back_populates="organization")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    org_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("organizations.id"), nullable=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(50), default="USER", index=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    organization: Mapped[Optional[Organization]] = relationship("Organization", back_populates="users")
    api_keys: Mapped[List[APIKey]] = relationship("APIKey", back_populates="user")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    organization: Mapped[Organization] = relationship("Organization", back_populates="projects")
    verifications: Mapped[List[VerificationSession]] = relationship("VerificationSession", back_populates="project")


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    rate_limit_rpm: Mapped[int] = mapped_column(Integer, default=600)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user: Mapped[User] = relationship("User", back_populates="api_keys")


class VerificationSession(Base):
    __tablename__ = "verification_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("projects.id"), nullable=True, index=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    hallucisense_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    risk_category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    execution_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    payload_json: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    project: Mapped[Optional[Project]] = relationship("Project", back_populates="verifications")

    __table_args__ = (
        Index("idx_verification_session_score_created", "hallucisense_score", "created_at"),
    )


class EvidenceCache(Base):
    __tablename__ = "evidence_cache"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    provider_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    evidence_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ProviderResponse(BaseModel):
    __tablename__ = "provider_responses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    provider_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status_code: Mapped[int] = mapped_column(Integer, default=200)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
