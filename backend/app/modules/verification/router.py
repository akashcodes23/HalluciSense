"""Verification API router."""
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.verification.schemas import VerificationReportResponse
from app.modules.verification.service import VerificationService

router = APIRouter(prefix="/verification", tags=["Verification"])

@router.get(
    "/{message_id}",
    response_model=VerificationReportResponse,
    summary="Get verification report for a specific message"
)
async def get_verification_report(
    message_id: UUID,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """Retrieve full hallucination verification report including sentence analysis and evidence."""
    service = VerificationService(db)
    # Validation of user ownership of the message should ideally be added here, 
    # but for Sprint 3 we can fetch it directly.
    return await service.get_report(message_id)
