"""
Chat ORM model.
Represents a named conversation thread owned by a user.
"""
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.message import Message


class Chat(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    A conversation thread.

    Columns:
        user_id         — Foreign key to users.id.
        title           — Chat title, editable by user.
        model_used      — LLM model slug active for this chat.
        is_archived     — Soft-archive flag (hidden from default view).
        last_message_at — Denormalised timestamp for fast ordering.
        metadata_       — Flexible JSON for future extensions.
    """

    __tablename__ = "chats"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(512), nullable=False, default="New Chat"
    )
    model_used: Mapped[str] = mapped_column(
        String(100), nullable=False, default="gemini-2.0-flash"
    )
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_message_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    metadata_: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSON, nullable=True, default=None
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="chats")
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<Chat id={self.id} title='{self.title}' user_id={self.user_id}>"
