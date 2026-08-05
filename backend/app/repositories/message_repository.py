"""
MessageRepository — database queries related to chat messages and verification data.
"""

from __future__ import annotations

from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.message import Message
from app.models.sentence_analysis import SentenceAnalysis
from app.models.verification_report import VerificationReport
from app.repositories.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """
    Repository responsible for all Message database operations.
    """

    model = Message

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def get_messages_by_chat_id(
        self,
        chat_id: UUID,
        limit: int | None = None,
        offset: int = 0,
        newest_first: bool = False,
        include_verification: bool = True,
    ) -> List[Message]:
        """
        Fetch messages belonging to a chat.

        Args:
            chat_id: Chat UUID.
            limit: Maximum number of messages to return.
            offset: Number of messages to skip.
            newest_first: Return newest messages first if True.
            include_verification: Eager-load verification tree.

        Returns:
            List[Message]
        """

        stmt = select(Message)

        if include_verification:
            stmt = stmt.options(
                selectinload(Message.verification_report).options(
                    selectinload(
                        VerificationReport.sentence_analyses
                    ).options(
                        selectinload(
                            SentenceAnalysis.evidence_items
                        )
                    )
                )
            )

        stmt = stmt.where(
            Message.chat_id == chat_id
        )

        stmt = stmt.order_by(
            Message.created_at.desc()
            if newest_first
            else Message.created_at.asc()
        )

        if offset > 0:
            stmt = stmt.offset(offset)

        if limit is not None:
            stmt = stmt.limit(limit)

        result = await self._session.execute(stmt)

        messages = list(result.scalars().unique().all())

        if newest_first:
            messages.reverse()

        return messages

    async def get_message_with_verification(
        self,
        message_id: UUID,
    ) -> Message | None:
        """
        Fetch a single message together with its complete
        verification hierarchy.
        """

        stmt = (
            select(Message)
            .options(
                selectinload(
                    Message.verification_report
                ).options(
                    selectinload(
                        VerificationReport.sentence_analyses
                    ).options(
                        selectinload(
                            SentenceAnalysis.evidence_items
                        )
                    )
                )
            )
            .where(Message.id == message_id)
        )

        result = await self._session.execute(stmt)

        return result.scalar_one_or_none()