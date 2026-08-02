"""Verification Schemas."""
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from app.core.constants import RiskLevel

class EvidenceItemResponse(BaseModel):
    id: UUID
    claim: str
    snippet: str
    source_name: str
    source_url: str
    similarity_score: float
    is_supporting: bool
    model_config = ConfigDict(from_attributes=True)

class SentenceAnalysisResponse(BaseModel):
    id: UUID
    sentence_index: int
    sentence_text: str
    h_score: float
    risk_level: str
    color_code: str
    factual_error: float
    confidence_gap: Optional[float] = None
    consistency_failure: Optional[float] = None
    reasoning: Optional[str] = None
    evidence: List[EvidenceItemResponse]
    model_config = ConfigDict(from_attributes=True)

class PillarAvailability(BaseModel):
    pillar1: bool = True
    pillar2: bool = False
    pillar3: bool = False

class VerificationReportResponse(BaseModel):
    id: UUID
    message_id: UUID
    overall_h_score: float
    overall_risk_level: str
    trust_score: Optional[float] = None
    factual_error_score: float
    confidence_gap_score: Optional[float] = None
    consistency_failure_score: Optional[float] = None
    weights_used: Dict[str, Any]
    pillar_availability: Optional[PillarAvailability] = None
    corrected_response: Optional[str] = None
    processing_time_ms: float
    sentence_analyses: List[SentenceAnalysisResponse]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
