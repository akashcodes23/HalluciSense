"""
Users Pydantic schemas.
"""
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


class UserUpdateRequest(BaseModel):
    """Payload for PATCH /users/me."""

    full_name: Optional[str] = Field(None, min_length=2, max_length=255)
    avatar_url: Optional[str] = Field(None, max_length=1024)
    preferred_model: Optional[str] = Field(None, max_length=100)


class UserDetailResponse(BaseModel):
    """Full user profile response."""

    id: str
    email: str
    full_name: str
    avatar_url: Optional[str]
    role: str
    preferred_model: str
    is_active: bool
    is_verified: bool

    model_config = {"from_attributes": True}
