"""
Chat Pydantic schemas.
"""
from datetime import datetime
from typing import List, Optional
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
