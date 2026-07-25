from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database.session import get_db
from app.models.user import User
from app.models.message import Message
from app.models.chat import Chat
from app.modules.auth.dependencies import get_current_user

router = APIRouter(prefix="/admin", tags=["Admin"])

async def get_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user

@router.get("/users")
async def list_users(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """
    List all users with their chat counts.
    """
    stmt = select(User).order_by(User.created_at.desc())
    res = await db.execute(stmt)
    users = res.scalars().all()
    
    result = []
    for user in users:
        # Get chat count
        c_stmt = select(func.count(Chat.id)).where(Chat.user_id == user.id)
        chat_count = await db.scalar(c_stmt) or 0
        
        result.append({
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "chat_count": chat_count
        })
    return result

@router.get("/system-health")
async def get_system_health(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """
    Get system health and stats.
    """
    u_stmt = select(func.count(User.id))
    total_users = await db.scalar(u_stmt) or 0
    
    m_stmt = select(func.count(Message.id))
    total_messages = await db.scalar(m_stmt) or 0
    
    return {
        "status": "healthy",
        "total_users": total_users,
        "total_messages": total_messages,
        "database": "connected"
    }
