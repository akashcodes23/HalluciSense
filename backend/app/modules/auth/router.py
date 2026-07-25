"""
Auth router — HTTP interface for all authentication endpoints.
Business logic is delegated entirely to AuthService.
"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AuthenticationError,
    ConflictError,
    TokenInvalidError,
)
from app.database.session import get_db
from app.modules.auth.dependencies import CurrentUser
from app.modules.auth.schemas import (
    AuthResponse,
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserPublicResponse,
)
from app.modules.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    body: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """
    Create a new HalluciSense account.
    Returns a token pair and basic user info on success.
    """
    try:
        service = AuthService(db)
        user, access_token, refresh_token = await service.register(
            full_name=body.full_name,
            email=body.email,
            password=body.password,
        )
    except ConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=exc.message
        )

    return AuthResponse(
        tokens=TokenResponse(
            access_token=access_token, refresh_token=refresh_token
        ),
        user=UserPublicResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            preferred_model=user.preferred_model,
            is_verified=user.is_verified,
        ),
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Authenticate and receive a token pair",
)
async def login(
    body: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """
    Authenticate with email and password.
    Returns an access token (15 min) and a refresh token (7 days).
    """
    try:
        service = AuthService(db)
        user, access_token, refresh_token = await service.login(
            email=body.email,
            password=body.password,
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.message
        )

    return AuthResponse(
        tokens=TokenResponse(
            access_token=access_token, refresh_token=refresh_token
        ),
        user=UserPublicResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            preferred_model=user.preferred_model,
            is_verified=user.is_verified,
        ),
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh token for a new token pair",
)
async def refresh_token(
    body: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """
    Validate the provided refresh token and issue a fresh access + refresh pair.
    """
    try:
        service = AuthService(db)
        _, access_token, new_refresh = await service.refresh(body.refresh_token)
    except (TokenInvalidError, AuthenticationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.message
        )

    return TokenResponse(access_token=access_token, refresh_token=new_refresh)


@router.get(
    "/me",
    response_model=UserPublicResponse,
    summary="Return the authenticated user's profile",
)
async def get_me(current_user: CurrentUser) -> UserPublicResponse:
    """Returns the currently authenticated user's public profile."""
    return UserPublicResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        preferred_model=current_user.preferred_model,
        is_verified=current_user.is_verified,
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Invalidate the current session",
)
async def logout(current_user: CurrentUser) -> MessageResponse:
    """
    Logout endpoint. In a full implementation this blacklists the
    refresh token in Redis. Stateless for Sprint 1.
    """
    return MessageResponse(message="Successfully logged out.")
