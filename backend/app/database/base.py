"""
SQLAlchemy declarative base, engine factory, and shared column mixins.
All models import Base from this module — never from sqlalchemy directly.
"""
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, String
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, AsyncEngine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings


# ---------------------------------------------------------------------------
# Declarative Base
# ---------------------------------------------------------------------------

class Base(AsyncAttrs, DeclarativeBase):
    """
    Shared declarative base for all ORM models.
    AsyncAttrs enables awaitable lazy-loads on async sessions.
    """
    pass


# ---------------------------------------------------------------------------
# Reusable Column Mixins
# ---------------------------------------------------------------------------

class UUIDPrimaryKeyMixin:
    """Provides a UUID primary key with server-side default."""

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )


class TimestampMixin:
    """Provides created_at and updated_at columns with UTC defaults."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
        onupdate=lambda: datetime.now(tz=timezone.utc),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Engine Factory
# ---------------------------------------------------------------------------

def create_engine() -> AsyncEngine:
    """
    Create an async SQLAlchemy engine from application settings.
    Uses pooling for PostgreSQL, StaticPool for SQLite.
    """
    url = settings.DATABASE_URL
    is_sqlite = url.startswith("sqlite")
    if is_sqlite:
        from sqlalchemy.pool import StaticPool
        return create_async_engine(
            url,
            echo=not settings.is_production,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_async_engine(
        url,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        echo=not settings.is_production,
    )
