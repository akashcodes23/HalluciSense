"""
HalluciSense Pillar 2 — Evidence Feature Generation Schemas
============================================================
Pydantic schemas for the 10 Pillar 2 evidence features.
"""

from typing import Dict
from pydantic import BaseModel, Field


class PillarTwoFeatures(BaseModel):
    support_ratio: float = Field(..., ge=0.0, le=1.0, description="Proportion of claims supported by evidence")
    contradiction_ratio: float = Field(..., ge=0.0, le=1.0, description="Proportion of claims contradicted by evidence")
    authority_score: float = Field(..., ge=0.0, le=1.0, description="Mean authority score of retrieved sources")
    source_diversity: float = Field(..., ge=0.0, le=1.0, description="Normalized entropy of evidence provider sources")
    evidence_coverage: float = Field(..., ge=0.0, le=1.0, description="Proportion of claims with at least 1 evidence item")
    evidence_density: float = Field(..., ge=0.0, description="Mean evidence items retrieved per claim")
    citation_quality: float = Field(..., ge=0.0, le=1.0, description="Proportion of evidence items with DOI or academic citation")
    consensus_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean consensus confidence score across claims")
    recency_score: float = Field(..., ge=0.0, le=1.0, description="Proportion of evidence published after 2020")
    verification_completeness: float = Field(..., ge=0.0, le=1.0, description="Overall verification pipeline execution completeness")

    def to_dict(self) -> Dict[str, float]:
        """Return dictionary representation of the 10 feature values."""
        return self.model_dump()
