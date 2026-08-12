"""HalluciSense v1.0 Production Pydantic Schemas.

Defines production-grade input and output models for POST /api/v1/analyze,
POST /api/v1/explain, and GET /api/v1/metrics matching OpenAPI 3.0 standards.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Input request model for canonical hallucination analysis."""
    query: Optional[str] = Field(None, description="User query or context prompt", example="Who invented the telephone?")
    response: str = Field(..., min_length=1, description="Generated LLM response to evaluate", example="Alexander Graham Bell invented the telephone in 1876.")
    model_name: Optional[str] = Field("GPT-4", description="Target LLM architecture name", example="GPT-4")


class ExplainRequest(BaseModel):
    """Input request model for detailed hallucination explainability."""
    query: Optional[str] = Field(None, description="User query or context prompt", example="Who invented the telephone?")
    response: str = Field(..., min_length=1, description="Generated LLM response to evaluate", example="Alexander Graham Bell invented the telephone in 1876.")
    model_name: Optional[str] = Field("GPT-4", description="Target LLM architecture name", example="GPT-4")


class PillarScores(BaseModel):
    """Pillar score breakdown (0.0 to 1.0 risk)."""
    retrieval: float = Field(..., ge=0.0, le=1.0, description="Pillar 1 Evidence Grounding risk score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pillar 2 Predictive Confidence risk score")
    consistency: float = Field(..., ge=0.0, le=1.0, description="Pillar 3 Structural Consistency risk score")


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
    trace_id: str = Field(..., description="Unique execution trace ID for pipeline debugging", example="TRACE_88CFA3E9")
    overall_h_score: float = Field(..., ge=0.0, le=1.0, description="Platt recalibrated hallucination risk H(q) in [0, 1]", example=0.08)
    risk_level: str = Field(..., description="Categorical risk level: VERIFIED, LOW_RISK, MODERATE_RISK, LIKELY_HALLUCINATED", example="VERIFIED")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall system confidence in classification [0, 1]", example=0.94)
    pillar_scores: PillarScores = Field(...)
    failure_taxonomy: str = Field(default="NONE", description="Single-label failure taxonomy category", example="NONE")
    processing_time_ms: float = Field(..., ge=0.0, description="Total pipeline execution latency in milliseconds", example=143.0)
    version: str = Field("1.0.0", description="HalluciSense production release version")
    hallucination: Optional[bool] = Field(False, description="Boolean flag indicating hallucinated output")
    sentence_scores: List[SentenceScore] = Field(default_factory=list)
    token_heatmap: List[TokenHeatmapItem] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    confidence_analysis: Optional[ConfidenceAnalysis] = Field(None)
    root_cause_classification: Optional[str] = Field("VERIFIED", description="Single-label failure classification")


class ExplainResponse(BaseModel):
    """Detailed explainability response model for POST /api/v1/explain."""
    trace_id: str = Field(..., description="Unique execution trace ID")
    overall_h_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(...)
    retrieved_evidence: List[EvidenceItem] = Field(default_factory=list)
    supporting_passages: List[str] = Field(default_factory=list)
    contradiction_evidence: List[str] = Field(default_factory=list)
    token_heatmap: List[TokenHeatmapItem] = Field(default_factory=list)
    sentence_scores: List[SentenceScore] = Field(default_factory=list)
    reasoning_chain: List[str] = Field(default_factory=list)
    fusion_contribution: Dict[str, float] = Field(default_factory=dict)
    adaptive_weights: Dict[str, float] = Field(default_factory=dict)
    confidence_explanation: str = Field(...)


class MetricsResponse(BaseModel):
    """Production metrics response model for GET /api/v1/metrics."""
    requests: int = Field(..., ge=0, description="Total requests processed", example=152)
    average_latency_ms: float = Field(..., ge=0.0, description="Average execution latency in ms", example=143.0)
    average_h_score: float = Field(..., ge=0.0, le=1.0, description="Average overall hallucination score", example=0.18)
    success_rate: float = Field(..., ge=0.0, le=100.0, description="Successful requests percentage", example=99.7)
    error_rate: float = Field(..., ge=0.0, le=100.0, description="Failed requests percentage", example=0.3)
    memory_mb: float = Field(..., ge=0.0, description="Process RSS RAM memory usage in MB", example=421.0)
