"""
ChatService — use cases for managing chat sessions.
"""
from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.chat import Chat
from app.repositories.chat_repository import ChatRepository


class ChatService:
    """Handles Chat CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = ChatRepository(session)

    async def get_user_chats(self, user_id: UUID, limit: int = 50, offset: int = 0) -> tuple[List[Chat], int]:
        """
        Fetch all active chats for a user.
        Returns a tuple of (chats list, total count).
        (For simplicity, we return length of list as total in this sprint, but ideally a count query should be used).
        """
        chats = await self._repo.get_by_user_id(user_id=user_id, limit=limit, offset=offset)
        return chats, len(chats)

    async def get_chat(self, chat_id: UUID, user_id: UUID) -> Chat:
        """
        Fetch a single chat, ensuring it belongs to the user.
        """
        chat = await self._repo.get_chat_with_messages(chat_id=chat_id, user_id=user_id)
        if chat is None:
            raise NotFoundError("Chat", str(chat_id))
        return chat

    async def create_chat(self, user_id: UUID, title: str, model_used: str) -> Chat:
        """
        Create a new chat session for the user.
        """
        chat = Chat(
            user_id=user_id,
            title=title,
            model_used=model_used
        )
        return await self._repo.create(chat)

    async def update_chat(
        self, chat_id: UUID, user_id: UUID, title: Optional[str] = None, is_archived: Optional[bool] = None
    ) -> Chat:
        """
        Update a chat's metadata (e.g., rename or archive).
        """
        chat = await self.get_chat(chat_id=chat_id, user_id=user_id)
        
        updates = {}
        if title is not None:
            updates["title"] = title
        if is_archived is not None:
            updates["is_archived"] = is_archived
            
        if updates:
            chat = await self._repo.update(chat, **updates)
            
        return chat

    async def delete_chat(self, chat_id: UUID, user_id: UUID) -> None:
        """
        Hard delete a chat.
        """
        chat = await self.get_chat(chat_id=chat_id, user_id=user_id)
        await self._repo.delete(chat)
