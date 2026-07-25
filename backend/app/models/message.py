"""
Message ORM model.
Represents a single turn in a chat conversation (user or AI).
"""
import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import MessageRole, VerificationStatus
from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.chat import Chat
    from app.models.user import User
    from app.models.verification_report import VerificationReport


class Message(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    A single message turn.

    Columns:
        chat_id             — Parent chat (FK).
        user_id             — Authoring user (FK). NULL for system messages.
        role                — USER | ASSISTANT | SYSTEM.
        content             — Full text of the message.
        raw_logits          — JSON array of {token, probability} captured from LLM.
                              NULL for user messages.
        processing_time_ms  — Time to receive full LLM completion.
        verification_status — PENDING → PROCESSING → COMPLETE | FAILED.
    """

    __tablename__ = "messages"

    chat_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default=MessageRole.USER
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    raw_logits: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    verification_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=VerificationStatus.PENDING,
        index=True,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    chat: Mapped["Chat"] = relationship("Chat", back_populates="messages")
    user: Mapped[Optional["User"]] = relationship("User")
    verification_report: Mapped[Optional["VerificationReport"]] = relationship(
        "VerificationReport",
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role} chat_id={self.chat_id}>"
