"""
UserRepository — all database queries related to users.
"""
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """
    Concrete repository for User entities.

    Provides specialised finders beyond the generic CRUD operations.
    """

    model = User

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Return an active user by email address (case-insensitive)."""
        result = await self._session.execute(
            select(User).where(
                User.email == email.lower().strip(),
                User.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def email_exists(self, email: str) -> bool:
        """Return True if any user (active or inactive) holds this email."""
        result = await self._session.execute(
            select(User.id).where(User.email == email.lower().strip())
        )
        return result.scalar_one_or_none() is not None

    async def get_active_by_id(self, user_id) -> Optional[User]:
        """Return an active user by primary key."""
        if isinstance(user_id, str):
            import uuid
            try:
                user_id = uuid.UUID(user_id)
            except ValueError:
                return None
                
        result = await self._session.execute(
            select(User).where(
                User.id == user_id,
                User.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()
