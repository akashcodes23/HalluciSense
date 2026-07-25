"""
Messages Pydantic schemas.
"""
from datetime import datetime
from typing import List, Optional, Any, Dict
from uuid import UUID
from pydantic import BaseModel, Field


class MessageCreateRequest(BaseModel):
    """Payload for POST /chats/{chat_id}/messages (for non-streaming chat)."""
    content: str = Field(..., min_length=1)


class EvidenceItemResponse(BaseModel):
    id: UUID
    claim: str
    snippet: str
    source_name: str
    source_url: Optional[str]
    similarity_score: float
    is_supporting: bool
    model_config = {"from_attributes": True}


class SentenceAnalysisResponse(BaseModel):
    id: UUID
    sentence_index: int
    sentence_text: str
    h_score: float
    risk_level: str
    color_code: str
    reasoning: Optional[str]
    evidence_items: List[EvidenceItemResponse] = []
    model_config = {"from_attributes": True}


class VerificationReportResponse(BaseModel):
    id: UUID
    overall_h_score: float
    overall_risk_level: str
    processing_time_ms: Optional[float]
    sentence_analyses: List[SentenceAnalysisResponse] = []
    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    """Standard message response."""
    id: UUID
    chat_id: UUID
    role: str
    content: str
    verification_status: str
    created_at: datetime
    verification_report: Optional[VerificationReportResponse] = None
    model_config = {"from_attributes": True}


class MessageListResponse(BaseModel):
    """Response for GET /chats/{chat_id}/messages."""
    items: List[MessageResponse]
    total: int


class StreamingTokenResponse(BaseModel):
    """Response yielded during WebSocket text generation stream."""
    type: str = "token"
    text: str


class StreamingVerificationResponse(BaseModel):
    """Response yielded when verification completes via WebSocket."""
    type: str = "verification"
    report: VerificationReportResponse
