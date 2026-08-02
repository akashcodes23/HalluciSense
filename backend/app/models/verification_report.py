"""
VerificationReport ORM model.
Stores the top-level hallucination analysis result for a single AI message.
Has a 1-to-1 relationship with Message.
"""
import uuid
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Float, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import RiskLevel
from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.message import Message
    from app.models.sentence_analysis import SentenceAnalysis


class VerificationReport(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Top-level hallucination report for one AI message.

    Columns:
        message_id              — 1-to-1 FK to messages.id.
        overall_h_score         — Aggregate H-Score ∈ [0.0, 1.0].
        overall_risk_level      — VERIFIED | NEEDS_VERIFICATION | LIKELY_HALLUCINATED.
        factual_error_score     — Pillar 1 FE score.
        confidence_gap_score    — Pillar 2 CG score.
        consistency_failure_score — Pillar 3 CF score.
        weights_used            — JSON {'alpha': 0.45, 'beta': 0.30, 'gamma': 0.25}.
        pillar1_summary         — JSON snapshot of Pillar1Result.
        pillar2_summary         — JSON snapshot of Pillar2Result.
        pillar3_summary         — JSON snapshot of Pillar3Result.
        processing_time_ms      — Total pipeline wall-clock time.
    """

    __tablename__ = "verification_reports"

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    overall_h_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_risk_level: Mapped[str] = mapped_column(
        String(30), nullable=False, default=RiskLevel.NEEDS_VERIFICATION, index=True
    )
    factual_error_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence_gap_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    consistency_failure_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weights_used: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    pillar1_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    pillar2_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    pillar3_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    corrected_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    message: Mapped["Message"] = relationship(
        "Message", back_populates="verification_report"
    )
    sentence_analyses: Mapped[List["SentenceAnalysis"]] = relationship(
        "SentenceAnalysis",
        back_populates="report",
        cascade="all, delete-orphan",
        order_by="SentenceAnalysis.sentence_index",
    )

    def __repr__(self) -> str:
        return (
            f"<VerificationReport id={self.id} "
            f"h_score={self.overall_h_score:.3f} "
            f"risk={self.overall_risk_level}>"
        )
