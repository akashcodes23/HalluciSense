"""
SentenceAnalysis ORM model.
Stores the per-sentence hallucination breakdown within a VerificationReport.
"""
import uuid
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import RiskLevel
from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.verification_report import VerificationReport
    from app.models.evidence_item import EvidenceItem


class SentenceAnalysis(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Per-sentence hallucination scores and metadata.

    Columns:
        report_id         — Parent VerificationReport (FK).
        sentence_index    — 0-based position in the original response.
        sentence_text     — Raw text of the sentence.
        start_char        — Character offset (start) in full response.
        end_char          — Character offset (end) in full response.
        h_score           — Fused H-Score for this sentence.
        risk_level        — VERIFIED | NEEDS_VERIFICATION | LIKELY_HALLUCINATED.
        color_code        — Hex colour string for UI rendering.
        factual_error     — Pillar 1 contribution.
        confidence_gap    — Pillar 2 contribution.
        consistency_failure — Pillar 3 contribution.
        reasoning         — Human-readable explanation.
    """

    __tablename__ = "sentence_analyses"

    report_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("verification_reports.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sentence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    sentence_text: Mapped[str] = mapped_column(Text, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    h_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(
        String(30), nullable=False, default=RiskLevel.NEEDS_VERIFICATION
    )
    color_code: Mapped[str] = mapped_column(String(10), nullable=False, default="#F59E0B")
    factual_error: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence_gap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    consistency_failure: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    report: Mapped["VerificationReport"] = relationship(
        "VerificationReport", back_populates="sentence_analyses"
    )
    evidence_items: Mapped[List["EvidenceItem"]] = relationship(
        "EvidenceItem",
        back_populates="sentence_analysis",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<SentenceAnalysis idx={self.sentence_index} "
            f"h={self.h_score:.3f} risk={self.risk_level}>"
        )
