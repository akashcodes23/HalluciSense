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
        NEEDS_VERIFICATION = "NEEDS_VERIFICATION"   # Yellow (#F59E0B)
        LIKELY_HALLUCINATED = "LIKELY_HALLUCINATED" # Red (#EF4444)

    class EvidenceItem(BaseModel):
        claim: str = Field(..., description="The factual claim extracted from the sentence")
        snippet: str = Field(..., description="Supporting or contradicting text snippet from source")
        source_name: str = Field(..., description="Name of knowledge source e.g. Wikipedia, Internal KB")
        source_url: Optional[str] = Field(None, description="URL or URI of the reference source")
        similarity_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score to claim")
        is_supporting: bool = Field(True, description="Whether the evidence supports or refutes the claim")

    class TokenAnalysis(BaseModel):
        token: str = Field(..., description="The raw token string")
        position: int = Field(..., ge=0, description="Token sequence index")
        probability: float = Field(..., ge=0.0, le=1.0, description="Logit softmax probability")
        entropy: float = Field(..., ge=0.0, description="Token entropy score")
        risk_level: RiskLevel = Field(..., description="Assigned risk categorization")
        color_code: str = Field(..., description="Hex or Tailwind color indicator")

    class Pillar1Result(BaseModel):
        claims: List[str] = Field(default_factory=list)
        evidence: List[EvidenceItem] = Field(default_factory=list)
        factual_error_score: float = Field(..., ge=0.0, le=1.0)
        reasoning: str = Field(...)

    class Pillar2Result(BaseModel):
        avg_probability: float = Field(..., ge=0.0, le=1.0)
        avg_entropy: float = Field(..., ge=0.0)
        confidence_gap_score: float = Field(..., ge=0.0, le=1.0)
        reasoning: str = Field(...)

    class Pillar3Result(BaseModel):
        sample_responses: List[str] = Field(default_factory=list)
        pairwise_similarities: List[float] = Field(default_factory=list)
        consistency_failure_score: float = Field(..., ge=0.0, le=1.0)
        reasoning: str = Field(...)

    class SentenceAnalysis(BaseModel):
        sentence_id: int = Field(...)
        text: str = Field(...)
        start_char: int = Field(...)
        end_char: int = Field(...)
        factual_error: float = Field(..., ge=0.0, le=1.0)
        confidence_gap: float = Field(..., ge=0.0, le=1.0)
        consistency_failure: float = Field(..., ge=0.0, le=1.0)
        hallucination_score: float = Field(..., ge=0.0, le=1.0)
        risk_level: RiskLevel
        color_code: str
        evidence: List[EvidenceItem] = Field(default_factory=list)
        reasoning: str = Field(...)

    class HallucinationReport(BaseModel):
        full_text: str = Field(...)
        overall_h_score: float = Field(..., ge=0.0, le=1.0)
        overall_risk_level: RiskLevel
        sentence_analyses: List[SentenceAnalysis] = Field(default_factory=list)
        token_analyses: List[TokenAnalysis] = Field(default_factory=list)
        pillar1_summary: Pillar1Result
        pillar2_summary: Pillar2Result
        pillar3_summary: Pillar3Result
        weights_used: Dict[str, float] = Field(...)
else:
    class RiskLevel(str, Enum):
        VERIFIED = "VERIFIED"
        NEEDS_VERIFICATION = "NEEDS_VERIFICATION"
        LIKELY_HALLUCINATED = "LIKELY_HALLUCINATED"

    @dataclass
    class EvidenceItem:
        claim: str
        snippet: str
        source_name: str
        similarity_score: float
        source_url: Optional[str] = None
        is_supporting: bool = True

    @dataclass
    class TokenAnalysis:
        token: str
        position: int
        probability: float
        entropy: float
        risk_level: RiskLevel
        color_code: str

    @dataclass
    class Pillar1Result:
        factual_error_score: float
        reasoning: str
        claims: List[str] = field(default_factory=list)
        evidence: List[EvidenceItem] = field(default_factory=list)

    @dataclass
    class Pillar2Result:
        avg_probability: float
        avg_entropy: float
        confidence_gap_score: float
        reasoning: str

    @dataclass
    class Pillar3Result:
        consistency_failure_score: float
        reasoning: str
        sample_responses: List[str] = field(default_factory=list)
        pairwise_similarities: List[float] = field(default_factory=list)

    @dataclass
    class SentenceAnalysis:
        sentence_id: int
        text: str
        start_char: int
        end_char: int
        factual_error: float
        confidence_gap: float
        consistency_failure: float
        hallucination_score: float
        risk_level: RiskLevel
        color_code: str
        reasoning: str
        evidence: List[EvidenceItem] = field(default_factory=list)

    @dataclass
    class HallucinationReport:
        full_text: str
        overall_h_score: float
        overall_risk_level: RiskLevel
        pillar1_summary: Pillar1Result
        pillar2_summary: Pillar2Result
        pillar3_summary: Pillar3Result
        weights_used: Dict[str, float]
        sentence_analyses: List[SentenceAnalysis] = field(default_factory=list)
        token_analyses: List[TokenAnalysis] = field(default_factory=list)
