"""HalluciSense v1.0 Production Pydantic Schemas.

Defines production-grade input and output models for POST /api/v1/analyze,
POST /api/v1/explain, and GET /api/v1/metrics matching OpenAPI 3.0 standards.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Input request model for canonical hallucination analysis."""
    query: Optional[str] = Field(None, max_length=2000, description="User query or context prompt", example="Who invented the telephone?")
    response: str = Field(..., min_length=1, max_length=10000, description="Generated LLM response to evaluate", example="Alexander Graham Bell invented the telephone in 1876.")
    model_name: Optional[str] = Field("GPT-4", description="Target LLM architecture name", example="GPT-4")
    provided_evidence: Optional[List[Any]] = Field(default=None, description="Optional ground-truth reference passages or evidence")
    sample_responses: Optional[List[str]] = Field(default=None, description="Optional alternate candidate generations for P3 consistency analysis")
    logprobs: Optional[List[float]] = Field(default=None, description="Optional token generation probabilities from model provider")


class ExplainRequest(BaseModel):
    """Input request model for detailed hallucination explainability."""
    query: Optional[str] = Field(None, description="User query or context prompt", example="Who invented the telephone?")
    response: str = Field(..., min_length=1, description="Generated LLM response to evaluate", example="Alexander Graham Bell invented the telephone in 1876.")
    model_name: Optional[str] = Field("GPT-4", description="Target LLM architecture name", example="GPT-4")
    provided_evidence: Optional[List[Any]] = Field(default=None, description="Optional ground-truth reference passages or evidence")
    sample_responses: Optional[List[str]] = Field(default=None, description="Optional alternate candidate generations")
    logprobs: Optional[List[float]] = Field(default=None, description="Optional token generation probabilities")


class PillarScores(BaseModel):
    """Pillar score breakdown (0.0 to 1.0 risk). Null when pillar is unavailable."""
    retrieval: float = Field(..., ge=0.0, le=1.0, description="Pillar 1 Evidence Grounding risk score")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Pillar 2 Predictive Confidence risk score (None if logprobs omitted)")
    consistency: Optional[float] = Field(None, ge=0.0, le=1.0, description="Pillar 3 Structural Consistency risk score (None if single generation)")


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


class MeasuredTimingBreakdown(BaseModel):
    """Real instrumented execution durations (measured with perf_counter). Never derived from weights."""
    retrieval_ms: Optional[float] = Field(None, ge=0.0, description="Real BM25 + dense passage retrieval latency in ms")
    bm25_ms: Optional[float] = Field(None, ge=0.0, description="BM25 lexical index search latency in ms")
    dense_ms: Optional[float] = Field(None, ge=0.0, description="FAISS/Embedding dense vector search latency in ms")
    nli_ms: Optional[float] = Field(None, ge=0.0, description="Cross-Encoder NLI inference latency in ms")
    gemini_generation_ms: Optional[float] = Field(None, ge=0.0, description="LLM provider API generation latency in ms")
    p1_latency_ms: float = Field(..., ge=0.0, description="Pillar 1 total measured execution latency in ms")
    p2_latency_ms: float = Field(..., ge=0.0, description="Pillar 2 total measured execution latency in ms")
    p3_latency_ms: float = Field(..., ge=0.0, description="Pillar 3 total measured execution latency in ms")
    fusion_latency_ms: float = Field(..., ge=0.0, description="Mathematical fusion computation latency in ms")
    total_latency_ms: float = Field(..., ge=0.0, description="End-to-end measured pipeline execution latency in ms")


class PillarExecutionStatus(BaseModel):
    """Explicit availability and execution status for each pillar."""
    p1_status: str = Field(default="EXECUTED", description="EXECUTED, DEGRADED, or FAILED")
    p2_status: str = Field(default="UNAVAILABLE", description="EXECUTED, UNAVAILABLE (no logprobs), PROXY, or SKIPPED")
    p3_status: str = Field(default="UNAVAILABLE", description="EXECUTED, UNAVAILABLE (single response), or SKIPPED")
    fusion_status: str = Field(default="FULL_THREE_PILLAR", description="FULL_THREE_PILLAR, PARTIAL_TWO_PILLAR, or PARTIAL_ONE_PILLAR")
    p1_available: bool = Field(default=True)
    p2_available: bool = Field(default=False)
    p3_available: bool = Field(default=False)
    is_full_analysis: bool = Field(default=False)


class MathematicalFusionDecomposition(BaseModel):
    """Mathematical provenance and linear contribution decomposition."""
    equation: str = Field(default="H = alpha*P1 + beta*P2 + gamma*P3")
    fusion_mode: str = Field(default="FULL_THREE_PILLAR", description="FULL_THREE_PILLAR or PARTIAL_RENORMALIZED")
    configured_weights: Dict[str, float] = Field(default_factory=lambda: {"alpha": 0.45, "beta": 0.30, "gamma": 0.25})
    effective_weights: Dict[str, float] = Field(default_factory=dict)
    pillar_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    weighted_contributions: Dict[str, Optional[float]] = Field(default_factory=dict)
    available_pillars: List[str] = Field(default_factory=list)
    missing_pillars: List[str] = Field(default_factory=list)
    uncalibrated_h_score: float = Field(..., ge=0.0, le=1.0)
    calibrated_h_score: float = Field(..., ge=0.0, le=1.0)
    is_full_analysis: bool = Field(default=False)
    explanation: Optional[str] = Field(None, description="Explicit human and machine-readable mathematical explanation")


class ConfidenceAnalysis(BaseModel):
    """White-box logit entropy and epistemic uncertainty analysis."""
    whitebox_entropy: Optional[float] = Field(None, ge=0.0)
    blackbox_variation_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    epistemic_uncertainty: Optional[float] = Field(None, ge=0.0, le=1.0)
    aleatoric_uncertainty: Optional[float] = Field(None, ge=0.0, le=1.0)
    methodology: str = Field(default="UNAVAILABLE", description="TOKEN_LOGPROBS, UNCERTAINTY_PROXY, or UNAVAILABLE")
    signal_type: str = Field(default="UNAVAILABLE", description="MEASURED, DERIVED, PROXY, or UNAVAILABLE")
    uncertainty_measure: Optional[str] = Field(None)
    generations_used: Optional[int] = Field(None)
    raw_signal_metadata: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[str] = Field(None)


class FeatureAttribution(BaseModel):
    """Local counterfactual attribution for one frozen hybrid feature."""
    index: int = Field(..., ge=0)
    feature: str = Field(...)
    value: float
    baseline_value: float
    counterfactual_probability: float = Field(..., ge=0.0, le=1.0)
    delta: float
    direction: str = Field(..., description="increases_hallucination, decreases_hallucination, or neutral")
    relative_strength: float = Field(..., ge=0.0, le=1.0)


class ModelExplainability(BaseModel):
    """Faithful local explanation of the frozen 19-feature hybrid classifier."""
    available: bool = Field(...)
    method: str = Field(...)
    methodology: Optional[str] = Field(None)
    baseline_method: Optional[str] = Field(None)
    baseline_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    observed_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    decision_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    decision_margin: Optional[float] = Field(None)
    interaction_gap: Optional[float] = Field(None)
    non_additivity_note: Optional[str] = Field(None)
    feature_count: Optional[int] = Field(None, ge=0)
    features: List[FeatureAttribution] = Field(default_factory=list)
    top_positive_drivers: List[FeatureAttribution] = Field(default_factory=list)
    top_negative_drivers: List[FeatureAttribution] = Field(default_factory=list)
    reason: Optional[str] = Field(None)


class AnalysisResponse(BaseModel):
    """Canonical production response model for POST /api/v1/analyze."""
    trace_id: str = Field(..., description="Unique execution trace ID for pipeline debugging", example="TRACE_88CFA3E9")
    overall_h_score: float = Field(..., ge=0.0, le=1.0, description="Platt recalibrated hallucination risk H(q) in [0, 1]", example=0.08)
    risk_level: str = Field(..., description="Categorical risk level: VERIFIED, NEEDS_VERIFICATION, MODERATE_RISK, LIKELY_HALLUCINATED", example="VERIFIED")
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
    measured_timings: Optional[MeasuredTimingBreakdown] = Field(None, description="Actual measured sub-operation timings")
    pillar_status: Optional[PillarExecutionStatus] = Field(None, description="Detailed execution and availability status")
    fusion_decomposition: Optional[MathematicalFusionDecomposition] = Field(None, description="Step-by-step linear algebra contribution breakdown")
    explainability: Optional[ModelExplainability] = Field(None, description="Faithful local feature attribution for the frozen hybrid classifier")


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
    fusion_decomposition: Optional[MathematicalFusionDecomposition] = Field(None)
    measured_timings: Optional[MeasuredTimingBreakdown] = Field(None)
    model_explainability: Optional[ModelExplainability] = Field(None, description="Faithful local feature attribution for the frozen hybrid classifier")


class MetricsResponse(BaseModel):
    """Production metrics response model for GET /api/v1/metrics."""
    requests: int = Field(..., ge=0, description="Total requests processed", example=152)
    average_latency_ms: Optional[float] = Field(None, ge=0.0, description="Average execution latency in ms (null if 0 requests)", example=143.0)
    average_h_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average overall hallucination score (null if 0 requests)", example=0.18)
    success_rate: Optional[float] = Field(None, ge=0.0, le=100.0, description="Successful requests percentage (null if 0 requests)", example=99.7)
    error_rate: Optional[float] = Field(None, ge=0.0, le=100.0, description="Failed requests percentage (null if 0 requests)", example=0.3)
    memory_mb: float = Field(..., ge=0.0, description="Process RSS RAM memory usage in MB", example=421.0)
    status: str = Field(default="READY")
