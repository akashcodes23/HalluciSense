"""
AnalyticsEvent ORM model.
Append-only event store for user interaction and system metrics.
"""
import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User


class AnalyticsEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Append-only event record.

    Event types (non-exhaustive):
        'message_sent'       — User sent a message.
        'verification_done'  — Analysis completed.
        'sentence_inspected' — User clicked a sentence.
        'chat_exported'      — Chat was exported.
        'model_switched'     — User changed model.

    Payload is flexible JSON keyed by event_type.
    """

    __tablename__ = "analytics_events"

    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped[Optional["User"]] = relationship(
        "User", back_populates="analytics_events"
    )

    def __repr__(self) -> str:
        return f"<AnalyticsEvent type={self.event_type} user_id={self.user_id}>"
