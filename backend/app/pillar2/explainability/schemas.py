"""
HalluciSense Pillar 2 — Explainability Engine Schemas
======================================================
Pydantic schemas for verification reports, source summaries, and risk recommendations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.pillar2.unified_hscore.schemas import RiskCategory


class SourceSummary(BaseModel):
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Canonical source URL")
    provider: str = Field(..., description="Provider name")
    authority_score: float = Field(..., description="Authority weighting")


class ClaimAnalysisItem(BaseModel):
    claim_id: str = Field(..., description="Target claim ID")
    claim_text: str = Field(..., description="Text of claim")
    consensus_label: str = Field(..., description="Consensus label string")
    confidence: float = Field(..., description="Consensus confidence")
    supporting_count: int = Field(..., description="Number of supporting sources")
    contradicting_count: int = Field(..., description="Number of contradicting sources")
    summary: str = Field(..., description="Per-claim verification summary")


class VerificationExplanation(BaseModel):
    executive_summary: str = Field(..., description="High-level human-readable narrative summary")
    claim_analysis: List[ClaimAnalysisItem] = Field(..., description="Detailed per-claim verification breakdown")
    evidence_analysis: str = Field(..., description="Overall analysis of retrieved evidence quality and coverage")
    supporting_sources: List[SourceSummary] = Field(default_factory=list, description="Top supporting reference sources")
    contradicting_sources: List[SourceSummary] = Field(default_factory=list, description="Top contradicting reference sources")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall verification confidence score")
    risk_category: RiskCategory = Field(..., description="Categorical risk classification")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="HalluciSense Score (0-100)")
    actionable_recommendations: List[str] = Field(..., description="List of clear, actionable system recommendations")
