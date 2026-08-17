"""
Chat Pydantic schemas for CRUD and Closed-Loop Verification + Correction.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class ChatCreateRequest(BaseModel):
    """Payload for POST /chats."""
    title: Optional[str] = Field(default="New Chat", max_length=512)
    model_used: Optional[str] = Field(default="gemini-2.0-flash", max_length=100)


class ChatUpdateRequest(BaseModel):
    """Payload for PATCH /chats/{chat_id}."""
    title: Optional[str] = Field(default=None, max_length=512)
    is_archived: Optional[bool] = Field(default=None)


class ChatResponse(BaseModel):
    """Standard chat response."""
    id: UUID
    user_id: UUID
    title: str
    model_used: str
    is_archived: bool
    last_message_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatListResponse(BaseModel):
    """Response for GET /chats."""
    items: List[ChatResponse]
    total: int


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 11: CLOSED-LOOP CHAT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ClosedLoopChatRequest(BaseModel):
    """Payload for POST /api/v1/chat."""
    message: str = Field(..., min_length=1, max_length=4000, description="User question or prompt")
    conversation_id: Optional[str] = Field(default=None, description="Optional UUID or thread ID")
    model_name: Optional[str] = Field(default="default", description="Generation LLM model name")
    enable_verification: bool = Field(default=True, description="Run HalluciSense verification pipeline")
    auto_correct: bool = Field(default=True, description="Automatically repair flagged hallucinations")


class VerificationSummary(BaseModel):
    status: str = Field(..., description="VERIFIED | CORRECTED | REVIEW | FAILED")
    h_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    claims_total: int
    claims_flagged: int


class CorrectionSummary(BaseModel):
    performed: bool
    reason: str
    claims_corrected: List[Dict[str, Any]] = Field(default_factory=list)
    original_to_corrected: List[Dict[str, str]] = Field(default_factory=list)


class ClosedLoopChatResponse(BaseModel):
    """Response returned by POST /api/v1/chat."""
    conversation_id: str
    message_id: str
    original_response: str
    final_response: str
    verification: VerificationSummary
    correction: CorrectionSummary
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    trace_id: str
    latency_ms: float
