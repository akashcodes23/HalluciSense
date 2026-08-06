"""HalluciSense v1.0 Production Pydantic Schemas.

Defines production-grade input and output models for POST /api/v1/analyze
matching OpenAPI 3.0 standards.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Input request model for canonical hallucination analysis."""
    query: str = Field(..., min_length=1, description="User query or context prompt", example="What is the capital of France?")
    response: str = Field(..., min_length=1, description="Generated LLM response to evaluate", example="The capital of France is Paris.")
    model_name: Optional[str] = Field("GPT-4", description="Target LLM architecture name", example="GPT-4")


class PillarScores(BaseModel):
    """Pillar score breakdown (0.0 to 1.0 risk)."""
    retrieval: float = Field(..., ge=0.0, le=1.0, description="Pillar 1 Evidence Grounding risk score (1 - FE)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pillar 2 Predictive Confidence risk score (1 - CG)")
    consistency: float = Field(..., ge=0.0, le=1.0, description="Pillar 3 Structural Consistency risk score (1 - CF)")


class SentenceScore(BaseModel):
    """Sentence-level risk breakdown."""
    sentence_index: int = Field(..., ge=0)
    text: str = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(...)


class TokenHeatmapItem(BaseModel):
    """Token-level 4-tier risk heatmap element."""
    token: str = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    tier: str = Field(..., description="Risk tier: GREEN, YELLOW, ORANGE, RED")
    color_hex: str = Field(..., example="#10B981")


class EvidenceItem(BaseModel):
    """Retrieved evidence passage citation."""
    id: str = Field(...)
    title: str = Field(...)
    snippet: str = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    source: str = Field(default="Wikipedia / BM25+Dense Index")


class ConfidenceAnalysis(BaseModel):
    """White-box logit entropy and epistemic uncertainty analysis."""
    whitebox_entropy: float = Field(..., ge=0.0)
    blackbox_variation_score: float = Field(..., ge=0.0, le=1.0)
    epistemic_uncertainty: float = Field(..., ge=0.0, le=1.0)
    aleatoric_uncertainty: float = Field(..., ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    """Canonical production response model for POST /api/v1/analyze."""
    overall_h_score: float = Field(..., ge=0.0, le=1.0, description="Platt recalibrated hallucination risk H(q) in [0, 1]")
    hallucination: bool = Field(..., description="Boolean flag indicating whether response contains hallucinations")
    risk_level: str = Field(..., description="Categorical risk level: Low, Moderate, High, Critical")
    pillar_scores: PillarScores = Field(...)
    sentence_scores: List[SentenceScore] = Field(default_factory=list)
    token_heatmap: List[TokenHeatmapItem] = Field(default_factory=list)
    failure_taxonomy: str = Field(default="Factual Contradiction / Factual Misattribution")
    evidence: List[EvidenceItem] = Field(default_factory=list)
    confidence_analysis: ConfidenceAnalysis = Field(...)
    processing_time_ms: float = Field(..., ge=0.0, description="Total pipeline execution latency in milliseconds")
    version: str = Field("1.0.0", description="HalluciSense production release version")
