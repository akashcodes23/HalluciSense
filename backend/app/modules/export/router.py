import json
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database.session import get_db
from app.models.chat import Chat
from app.models.message import Message
from app.modules.auth.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/export", tags=["Export"])

@router.get("/chats/{chat_id}")
async def export_chat(
    chat_id: UUID,
    format: str = "json",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Export a chat in JSON or Markdown format.
    """
    stmt = (
        select(Chat)
        .options(
            selectinload(Chat.messages).selectinload(Message.verification_report)
        )
        .where(Chat.id == chat_id, Chat.user_id == current_user.id)
    )
    chat = await db.scalar(stmt)
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    # Sort messages by created_at
    messages = sorted(chat.messages, key=lambda m: m.created_at)

    if format.lower() == "json":
        data = {
            "chat_id": str(chat.id),
            "title": chat.title,
            "created_at": chat.created_at.isoformat(),
            "messages": []
        }
        for msg in messages:
            msg_data = {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            if msg.verification_report:
                msg_data["verification"] = {
                    "h_score": msg.verification_report.overall_h_score,
                    "risk_level": msg.verification_report.overall_risk_level,
                }
            data["messages"].append(msg_data)
            
        return Response(content=json.dumps(data, indent=2), media_type="application/json")
        
    elif format.lower() == "md":
        lines = [f"# {chat.title}\n"]
        for msg in messages:
            role_name = "User" if msg.role == "USER" else "HalluciSense Engine"
            lines.append(f"### {role_name}\n")
            lines.append(f"{msg.content}\n")
            
            if msg.verification_report:
                lines.append(f"> [!NOTE]\n> Verification Result: {msg.verification_report.overall_risk_level} (H-Score: {msg.verification_report.overall_h_score:.2f})\n")
            
            lines.append("---\n")
            
        md_content = "\n".join(lines)
        return Response(content=md_content, media_type="text/markdown")
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'json' or 'md'")
