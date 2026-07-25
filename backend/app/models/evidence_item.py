"""
EvidenceItem ORM model.
Stores a single retrieved evidence snippet supporting or refuting a claim.
"""
import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.sentence_analysis import SentenceAnalysis


class EvidenceItem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    A single retrieved evidence snippet for a sentence claim.

    Columns:
        sentence_analysis_id — Parent SentenceAnalysis (FK).
        claim                — Extracted factual claim text.
        snippet              — Source text excerpt.
        source_name          — e.g. "Wikipedia: Paris".
        source_url           — Optional URL.
        similarity_score     — Cosine/cross-encoder score ∈ [0, 1].
        is_supporting        — True = supports claim, False = refutes.
    """

    __tablename__ = "evidence_items"

    sentence_analysis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sentence_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    snippet: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(String(512), nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_supporting: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    sentence_analysis: Mapped["SentenceAnalysis"] = relationship(
        "SentenceAnalysis", back_populates="evidence_items"
    )

    def __repr__(self) -> str:
        return (
            f"<EvidenceItem source='{self.source_name}' "
            f"sim={self.similarity_score:.3f} supporting={self.is_supporting}>"
        )
