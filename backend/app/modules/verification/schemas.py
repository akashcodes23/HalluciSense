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
    confidence_gap: float
    consistency_failure: float
    evidence: List[EvidenceItemResponse]
    model_config = ConfigDict(from_attributes=True)

class VerificationReportResponse(BaseModel):
    id: UUID
    message_id: UUID
    overall_h_score: float
    overall_risk_level: str
    factual_error_score: float
    confidence_gap_score: float
    consistency_failure_score: float
    weights_used: Dict[str, Any]
    processing_time_ms: float
    sentence_analyses: List[SentenceAnalysisResponse]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
