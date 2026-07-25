from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database.session import get_db
from app.models.message import Message
from app.models.verification_report import VerificationReport
from app.modules.auth.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/summary")
async def get_analytics_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns high-level analytics for the current user's chats.
    """
    # 1. Total messages verified
    stmt = (
        select(func.count(VerificationReport.id))
        .join(Message, VerificationReport.message_id == Message.id)
        .where(Message.user_id == current_user.id)
    )
    total_verified = await db.scalar(stmt) or 0

    # 2. Average H-Score
    stmt_h = (
        select(func.avg(VerificationReport.overall_h_score))
        .join(Message, VerificationReport.message_id == Message.id)
        .where(Message.user_id == current_user.id)
    )
    avg_h_score = await db.scalar(stmt_h) or 0.0

    # 3. Risk Level Distribution
    stmt_dist = (
        select(VerificationReport.overall_risk_level, func.count(VerificationReport.id))
        .join(Message, VerificationReport.message_id == Message.id)
        .where(Message.user_id == current_user.id)
        .group_by(VerificationReport.overall_risk_level)
    )
    dist_res = await db.execute(stmt_dist)
    risk_distribution = {row[0]: row[1] for row in dist_res.all()}

    # 4. Trend (last 10 verified messages)
    stmt_trend = (
        select(VerificationReport.overall_h_score, VerificationReport.created_at)
        .join(Message, VerificationReport.message_id == Message.id)
        .where(Message.user_id == current_user.id)
        .order_by(VerificationReport.created_at.desc())
        .limit(10)
    )
    trend_res = await db.execute(stmt_trend)
    trend_data = [{"h_score": row[0], "date": row[1].isoformat()} for row in reversed(trend_res.all())]

    return {
        "total_verified": total_verified,
        "avg_h_score": float(avg_h_score),
        "risk_distribution": risk_distribution,
        "trend_data": trend_data
    }
