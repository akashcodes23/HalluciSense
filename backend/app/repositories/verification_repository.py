"""
Verification Repository.
Handles fetching Verification Reports and Evidence from PostgreSQL.
"""
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.sentence_analysis import SentenceAnalysis
from app.models.verification_report import VerificationReport
from app.repositories.base import BaseRepository

class VerificationRepository(BaseRepository[VerificationReport]):
    model = VerificationReport

    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_report_by_message_id(self, message_id: UUID) -> VerificationReport | None:
        """Fetch the complete verification report for a message with all nested data."""
        stmt = (
            select(VerificationReport)
            .where(VerificationReport.message_id == message_id)
            .options(
                selectinload(VerificationReport.sentence_analyses)
                .selectinload(SentenceAnalysis.evidence_items)
            )
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()
