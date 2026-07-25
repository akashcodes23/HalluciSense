"""
UserService — use cases for managing user profiles.
"""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.user import User
from app.repositories.user_repository import UserRepository


class UserService:
    """Handles profile reads and updates."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = UserRepository(session)

    async def get_profile(self, user_id) -> User:
        """
        Fetch an active user by ID.
        Raises NotFoundError if not found.
        """
        user = await self._repo.get_active_by_id(user_id)
        if user is None:
            raise NotFoundError("User", user_id)
        return user

    async def update_profile(
        self,
        user: User,
        full_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ) -> User:
        """
        Apply partial updates to the user profile.
        Only non-None values are updated.
        """
        updates: dict = {}
        if full_name is not None:
            updates["full_name"] = full_name
        if avatar_url is not None:
            updates["avatar_url"] = avatar_url
        if preferred_model is not None:
            updates["preferred_model"] = preferred_model

        if updates:
            user = await self._repo.update(user, **updates)
        return user
