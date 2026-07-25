"""
ChatRepository — all database queries related to chats.
"""
from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.chat import Chat
from app.repositories.base import BaseRepository


class ChatRepository(BaseRepository[Chat]):
    """
    Concrete repository for Chat entities.
    """

    model = Chat

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_by_user_id(self, user_id: UUID, limit: int = 50, offset: int = 0) -> List[Chat]:
        """
        Fetch all active chats for a user, ordered by most recent message.
        """
        result = await self._session.execute(
            select(Chat)
            .where(
                Chat.user_id == user_id,
                Chat.is_archived.is_(False)
            )
            .order_by(Chat.last_message_at.desc().nulls_last(), Chat.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_chat_with_messages(self, chat_id: UUID, user_id: UUID) -> Chat | None:
        """
        Fetch a chat and eagerly load its messages if the user owns it.
        """
        result = await self._session.execute(
            select(Chat)
            .options(selectinload(Chat.messages))
            .where(
                Chat.id == chat_id,
                Chat.user_id == user_id,
                Chat.is_archived.is_(False)
            )
        )
        return result.scalar_one_or_none()
