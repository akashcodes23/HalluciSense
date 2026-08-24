from enum import Enum
from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    USE_PYDANTIC = True
except ImportError:
    USE_PYDANTIC = False
    from dataclasses import dataclass, field

if USE_PYDANTIC:
    class RiskLevel(str, Enum):
        VERIFIED = "VERIFIED"                       # Green (#10B981)
        LOW_RISK = "LOW_RISK"                       # Green (#10B981)
        NEEDS_VERIFICATION = "NEEDS_VERIFICATION"   # Yellow (#F59E0B)
        NEEDS_REVIEW = "NEEDS_REVIEW"               # Yellow (#F59E0B)
        MODERATE_RISK = "MODERATE_RISK"             # Orange (#F97316)
        LIKELY_HALLUCINATED = "LIKELY_HALLUCINATED" # Red (#EF4444)
        HALLUCINATED = "HALLUCINATED"               # Red (#EF4444) backward compatibility
        INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE" # Gray (#6B7280)
        ABSTAIN = "ABSTAIN"                         # Gray (#6B7280)

    class EvidenceItem(BaseModel):
        claim: str = Field(..., description="The factual claim extracted from the sentence")
        snippet: str = Field(..., description="Supporting or contradicting text snippet from source")
        source_name: str = Field(..., description="Name of knowledge source e.g. Wikipedia, Internal KB")
        source_url: Optional[str] = Field(None, description="URL or URI of the reference source")
        similarity_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score to claim")
        is_supporting: bool = Field(True, description="Whether the evidence supports or refutes the claim")
        citation_confidence: Optional[float] = Field(None, description="Confidence score of citation alignment")
        retrieved_passage: Optional[str] = Field(None, description="Full passage content retrieved")

    class TokenAnalysis(BaseModel):
        token: str = Field(..., description="The raw token string")
        position: int = Field(..., ge=0, description="Token sequence index")
        probability: float = Field(..., ge=0.0, le=1.0, description="Logit softmax probability")
        entropy: float = Field(..., ge=0.0, description="Token entropy score")
        risk_level: RiskLevel = Field(..., description="Assigned risk categorization")
        color_code: str = Field(..., description="Hex or Tailwind color indicator")
        attribution_score: Optional[float] = Field(None, description="Score propagated from sentence H-Score")

    class Pillar1Result(BaseModel):
        claims: List[str] = Field(default_factory=list)
        evidence: List[EvidenceItem] = Field(default_factory=list)
        factual_error_score: float = Field(..., ge=0.0, le=1.0)
        reasoning: str = Field(...)
        retrieved_passages: List[str] = Field(default_factory=list)
        citation_confidence_score: Optional[float] = Field(None)
        dense_retrieval_score: Optional[float] = Field(None)
        bm25_retrieval_score: Optional[float] = Field(None)
        cross_encoder_score: Optional[float] = Field(None)
        last_timings: Optional[Dict[str, Any]] = Field(default=None)

    class Pillar2Result(BaseModel):
        avg_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
        avg_entropy: Optional[float] = Field(None, ge=0.0)
        confidence_gap_score: Optional[float] = Field(None, ge=0.0, le=1.0)
        available: bool = Field(default=False)
        status: str = Field(default="UNAVAILABLE")
        reasoning: str = Field(...)
        # White-Box Metrics
        token_logprobs: List[float] = Field(default_factory=list)
        attention_entropy: Optional[float] = Field(None)
        predictive_entropy: Optional[float] = Field(None)
        mutual_information: Optional[float] = Field(None)
        epistemic_uncertainty: Optional[float] = Field(None)
        aleatoric_uncertainty: Optional[float] = Field(None)
        # Black-Box API Metrics
        top_k_logprob_diff: Optional[float] = Field(None)
        response_variance: Optional[float] = Field(None)
        calibration_score: Optional[float] = Field(None)
        last_timings: Optional[Dict[str, Any]] = Field(default=None)

    class NLIAnalysis(BaseModel):
        primary_claim: str
        comparison_claim: str
        semantic_similarity: float
        entailment_probability: Optional[float] = None
        neutral_probability: Optional[float] = None
        contradiction_probability: Optional[float] = None
        label: Optional[str] = "unavailable"
        nli_available: bool = False

    class Pillar3Result(BaseModel):
        sample_responses: List[str] = Field(default_factory=list)
        pairwise_similarities: List[float] = Field(default_factory=list)
        consistency_failure_score: Optional[float] = Field(None, ge=0.0, le=1.0)
        similarity_method: str = Field(default="unavailable")
        nli_analyses: List[NLIAnalysis] = Field(default_factory=list)
        contradiction_score: Optional[float] = Field(None)
        nli_available: bool = Field(default=False)
        alignment_method: str = Field(default="sentence_semantic_alignment")
        reasoning: str = Field(...)
        available: bool = Field(default=False)
        status: str = Field(default="UNAVAILABLE")
        paraphrase_matrix: List[List[float]] = Field(default_factory=list)
        sentence_consistency_score: Optional[float] = Field(None)
        last_timings: Optional[Dict[str, Any]] = Field(default=None)

    class SentenceAnalysis(BaseModel):
        sentence_id: int = Field(...)
        text: str = Field(...)
        start_char: int = Field(...)
        end_char: int = Field(...)
        factual_error: float = Field(..., ge=0.0, le=1.0)
        confidence_gap: Optional[float] = Field(None, ge=0.0, le=1.0)
        confidence_gap_status: str = Field(default="UNAVAILABLE")
        consistency_failure: Optional[float] = Field(None, ge=0.0, le=1.0)
        consistency_failure_status: str = Field(default="UNAVAILABLE")
        hallucination_score: float = Field(..., ge=0.0, le=1.0)
        risk_level: RiskLevel
        color_code: str
        evidence: List[EvidenceItem] = Field(default_factory=list)
        reasoning: str = Field(...)
        corrected_response: Optional[str] = Field(None)
        span_localization: Optional[Dict[str, Any]] = Field(None, description="Character and token-level span alignment")
        confidence_decomposition: Optional[Dict[str, float]] = Field(None, description="Decomposition of aleatoric vs epistemic confidence")

    class HallucinationReport(BaseModel):
        full_text: str = Field(...)
        corrected_response: Optional[str] = Field(None)
        overall_h_score: float = Field(..., ge=0.0, le=1.0)
        overall_risk_level: RiskLevel
        sentence_analyses: List[SentenceAnalysis] = Field(default_factory=list)
        token_analyses: List[TokenAnalysis] = Field(default_factory=list)
        pillar1_summary: Pillar1Result
        pillar2_summary: Pillar2Result
        pillar3_summary: Pillar3Result
        weights_used: Dict[str, float] = Field(...)
        validation_status: str = Field(default="VALIDATED_ZERO_NAN")
        confidence_decomposition: Optional[Dict[str, float]] = Field(None, description="Pillar confidence contribution breakdown")
        uncertainty_analysis: Optional[Dict[str, Any]] = Field(None, description="Epistemic and aleatoric uncertainty metrics")
        evidence_citations: List[Dict[str, Any]] = Field(default_factory=list, description="Claim-level evidence citations and URLs")
        calibrated_probability: Optional[float] = Field(None, description="Platt-scaled calibrated hallucination probability")
        fusion_mode: str = Field(default="ADAPTIVE", description="STATIC, ADAPTIVE, or GRADIENT weight mode")
        sensitivity_analysis: Optional[Dict[str, Any]] = Field(None, description="Weight sensitivity diagnostics grid")
        performance_timings: Optional[Dict[str, Any]] = Field(None, description="High-resolution latency timing breakdown")
else:
    class RiskLevel(str, Enum):
        VERIFIED = "VERIFIED"
        LOW_RISK = "LOW_RISK"
        NEEDS_VERIFICATION = "NEEDS_VERIFICATION"
        NEEDS_REVIEW = "NEEDS_REVIEW"
        MODERATE_RISK = "MODERATE_RISK"
        LIKELY_HALLUCINATED = "LIKELY_HALLUCINATED"
        HALLUCINATED = "HALLUCINATED"
        INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
        ABSTAIN = "ABSTAIN"

    @dataclass
    class EvidenceItem:
        claim: str
        snippet: str
        source_name: str
        similarity_score: float
        source_url: Optional[str] = None
        is_supporting: bool = True
        citation_confidence: Optional[float] = None
        retrieved_passage: Optional[str] = None

    @dataclass
    class TokenAnalysis:
        token: str
        position: int
        probability: float
        entropy: float
        risk_level: RiskLevel
        color_code: str
        attribution_score: Optional[float] = None

    @dataclass
    class Pillar1Result:
        factual_error_score: float
        reasoning: str
        claims: List[str] = field(default_factory=list)
        evidence: List[EvidenceItem] = field(default_factory=list)
        retrieved_passages: List[str] = field(default_factory=list)
        citation_confidence_score: Optional[float] = None
        dense_retrieval_score: Optional[float] = None
        bm25_retrieval_score: Optional[float] = None
        cross_encoder_score: Optional[float] = None

    @dataclass
    class Pillar2Result:
        avg_probability: Optional[float]
        avg_entropy: Optional[float]
        confidence_gap_score: Optional[float]
        reasoning: str
        available: bool = False
        status: str = "UNAVAILABLE"
        token_logprobs: List[float] = field(default_factory=list)
        attention_entropy: Optional[float] = None
        predictive_entropy: Optional[float] = None
        mutual_information: Optional[float] = None
        epistemic_uncertainty: Optional[float] = None
        aleatoric_uncertainty: Optional[float] = None
        top_k_logprob_diff: Optional[float] = None
        response_variance: Optional[float] = None
        calibration_score: Optional[float] = None

    @dataclass
    class NLIAnalysis:
        primary_claim: str
        comparison_claim: str
        semantic_similarity: float
        entailment_probability: Optional[float] = None
        neutral_probability: Optional[float] = None
        contradiction_probability: Optional[float] = None
        label: Optional[str] = "unavailable"
        nli_available: bool = False

    @dataclass
    class Pillar3Result:
        consistency_failure_score: Optional[float]
        reasoning: str
        sample_responses: List[str] = field(default_factory=list)
        pairwise_similarities: List[float] = field(default_factory=list)
        similarity_method: str = "unavailable"
        nli_analyses: List[NLIAnalysis] = field(default_factory=list)
        contradiction_score: Optional[float] = None
        nli_available: bool = False
        alignment_method: str = "sentence_semantic_alignment"
        available: bool = False
        status: str = "UNAVAILABLE"
        paraphrase_matrix: List[List[float]] = field(default_factory=list)
        sentence_consistency_score: Optional[float] = None

    @dataclass
    class SentenceAnalysis:
        sentence_id: int
        text: str
        start_char: int
        end_char: int
        factual_error: float
        confidence_gap: Optional[float]
        consistency_failure: Optional[float]
        hallucination_score: float
        risk_level: RiskLevel
        color_code: str
        reasoning: str
        confidence_gap_status: str = "UNAVAILABLE"
        consistency_failure_status: str = "UNAVAILABLE"
        evidence: List[EvidenceItem] = field(default_factory=list)
        corrected_response: Optional[str] = None
        span_localization: Optional[Dict[str, Any]] = None
        confidence_decomposition: Optional[Dict[str, float]] = None

    @dataclass
    class HallucinationReport:
        full_text: str
        corrected_response: Optional[str]
        overall_h_score: float
        overall_risk_level: RiskLevel
        pillar1_summary: Pillar1Result
        pillar2_summary: Pillar2Result
        pillar3_summary: Pillar3Result
        weights_used: Dict[str, float]
        sentence_analyses: List[SentenceAnalysis] = field(default_factory=list)
        token_analyses: List[TokenAnalysis] = field(default_factory=list)
        validation_status: str = "VALIDATED_ZERO_NAN"
        confidence_decomposition: Optional[Dict[str, float]] = None
        uncertainty_analysis: Optional[Dict[str, Any]] = None
        evidence_citations: List[Dict[str, Any]] = field(default_factory=list)
        calibrated_probability: Optional[float] = None
        fusion_mode: str = "ADAPTIVE"
        sensitivity_analysis: Optional[Dict[str, Any]] = None
