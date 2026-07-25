"""
User ORM model.
Represents a HalluciSense account. Supports email/password and OAuth.
"""
import uuid
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import UserRole
from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.chat import Chat
    from app.models.analytics_event import AnalyticsEvent


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Core user entity.

    Columns:
        id           — UUID primary key.
        email        — Unique, indexed email address.
        hashed_password — Bcrypt hash. NULL for OAuth-only accounts.
        full_name    — Display name.
        avatar_url   — Optional profile picture URL.
        role         — 'USER' or 'ADMIN'.
        preferred_model — LLM model slug chosen by the user.
        is_active    — False = soft-deleted or suspended.
        is_verified  — Email verification flag.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(320), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    avatar_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default=UserRole.USER, index=True
    )
    preferred_model: Mapped[str] = mapped_column(
        String(100), nullable=False, default="gemini-2.0-flash"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # ── Relationships ─────────────────────────────────────────────────────────
    chats: Mapped[List["Chat"]] = relationship(
        "Chat", back_populates="user", cascade="all, delete-orphan", lazy="select"
    )
    analytics_events: Mapped[List["AnalyticsEvent"]] = relationship(
        "AnalyticsEvent", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"
