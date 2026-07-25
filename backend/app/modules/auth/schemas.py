"""
Auth Pydantic schemas — request/response contracts for all auth endpoints.
"""
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    """Payload for POST /auth/register."""

    full_name: str = Field(..., min_length=2, max_length=255, examples=["Jane Smith"])
    email: EmailStr = Field(..., examples=["jane@example.com"])
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        examples=["StrongP@ssw0rd!"],
    )

    @field_validator("email", mode="before")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit.")
        return v


class LoginRequest(BaseModel):
    """Payload for POST /auth/login."""

    email: EmailStr = Field(..., examples=["jane@example.com"])
    password: str = Field(..., min_length=1, examples=["StrongP@ssw0rd!"])

    @field_validator("email", mode="before")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.lower().strip()


class RefreshRequest(BaseModel):
    """Payload for POST /auth/refresh."""

    refresh_token: str = Field(..., description="Long-lived refresh JWT.")


class TokenResponse(BaseModel):
    """Returned after a successful login or token refresh."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserPublicResponse(BaseModel):
    """Minimal user representation returned after registration or login."""

    id: str
    email: str
    full_name: str
    role: str
    preferred_model: str
    is_verified: bool

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    """Combines token pair with user info in a single login response."""

    tokens: TokenResponse
    user: UserPublicResponse


class MessageResponse(BaseModel):
    """Generic success message response."""

    message: str
