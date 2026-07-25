"""Verification Service."""
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.verification_repository import VerificationRepository
from app.models.verification_report import VerificationReport
from app.core.exceptions import NotFoundError

class VerificationService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = VerificationRepository(session)

    async def get_report(self, message_id: UUID) -> VerificationReport:
        report = await self.repo.get_report_by_message_id(message_id)
        if not report:
            raise NotFoundError(f"Verification report for message {message_id} not found.")
        return report
