"""
MessageRepository — all database queries related to messages and their verifications.
"""
from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.message import Message
from app.repositories.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """
    Concrete repository for Message entities.
    """

    model = Message

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_messages_by_chat_id(self, chat_id: UUID) -> List[Message]:
        """
        Fetch all messages for a given chat, ordered by creation time.
        Eagerly loads the verification report if present.
        """
        result = await self._session.execute(
            select(Message)
            .options(
                selectinload(Message.verification_report)
                .selectinload(Message.verification_report.class_.sentence_analyses)
                .selectinload(Message.verification_report.class_.sentence_analyses.property.mapper.class_.evidence_items)
            )
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
        )
        return list(result.scalars().all())

    async def get_message_with_verification(self, message_id: UUID) -> Message | None:
        """
        Fetch a specific message and eagerly load its full verification tree.
        """
        result = await self._session.execute(
            select(Message)
            .options(
                selectinload(Message.verification_report)
                .selectinload(Message.verification_report.class_.sentence_analyses)
                .selectinload(Message.verification_report.class_.sentence_analyses.property.mapper.class_.evidence_items)
            )
            .where(Message.id == message_id)
        )
        return result.scalar_one_or_none()
