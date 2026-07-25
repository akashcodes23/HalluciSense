"""
Users router — authenticated user profile endpoints.
"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.database.session import get_db
from app.modules.auth.dependencies import CurrentUser
from app.modules.users.schemas import UserDetailResponse, UserUpdateRequest
from app.modules.users.service import UserService

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/me",
    response_model=UserDetailResponse,
    summary="Get the authenticated user's full profile",
)
async def get_my_profile(current_user: CurrentUser) -> UserDetailResponse:
    """Returns the full profile of the authenticated user."""
    return UserDetailResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        avatar_url=current_user.avatar_url,
        role=current_user.role,
        preferred_model=current_user.preferred_model,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
    )


@router.patch(
    "/me",
    response_model=UserDetailResponse,
    summary="Update the authenticated user's profile",
)
async def update_my_profile(
    body: UserUpdateRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserDetailResponse:
    """
    Partially update the user's display name, avatar, or preferred model.
    Only provided fields are updated.
    """
    service = UserService(db)
    updated = await service.update_profile(
        user=current_user,
        full_name=body.full_name,
        avatar_url=body.avatar_url,
        preferred_model=body.preferred_model,
    )
    return UserDetailResponse(
        id=str(updated.id),
        email=updated.email,
        full_name=updated.full_name,
        avatar_url=updated.avatar_url,
        role=updated.role,
        preferred_model=updated.preferred_model,
        is_active=updated.is_active,
        is_verified=updated.is_verified,
    )
