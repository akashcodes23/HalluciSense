"""
Messages router — HTTP interface for Chat Messages and WebSockets.
"""
import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import AsyncSessionLocal, get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.messages.schemas import (
    MessageCreateRequest,
    MessageListResponse,
    MessageResponse,
)
from app.modules.messages.service import MessageService
from app.core.security import decode_token

router = APIRouter(prefix="/chats/{chat_id}/messages", tags=["Messages"])


@router.get(
    "",
    response_model=MessageListResponse,
    summary="List all messages and their verification reports for a chat",
)
async def list_messages(
    chat_id: UUID,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageListResponse:
    """Fetch all messages for a specific chat."""
    service = MessageService(db)
    messages, total = await service.get_chat_messages(chat_id, current_user.id)
    return MessageListResponse(
        items=[MessageResponse.model_validate(m) for m in messages],
        total=total,
    )


@router.websocket("/stream")
async def stream_chat_messages(
    websocket: WebSocket,
    chat_id: UUID,
    token: str = Query(..., description="JWT Bearer token for authentication")
):
    """
    WebSocket endpoint for streaming LLM generation and running the Verification Pipeline.
    Expects client to send a JSON message: {"content": "User prompt text"}
    """
    await websocket.accept()

    try:
        # Validate WebSocket token manually
        payload = decode_token(token)
        user_id_str = payload.get("sub")
        if not user_id_str:
            await websocket.close(code=1008, reason="Invalid token")
            return
        user_id = UUID(user_id_str)
        
        # Wait for user input
        data = await websocket.receive_text()
        parsed_data = json.loads(data)
        user_content = parsed_data.get("content")

        if not user_content:
            await websocket.close(code=1003, reason="Content missing")
            return

        async with AsyncSessionLocal() as session:
            service = MessageService(session)
            
            # Stream tokens to client
            async for chunk in service.stream_reply(chat_id, user_id, user_content):
                await websocket.send_json(chunk)

            # Close normally after verification completes
            await websocket.close(code=1000, reason="Completed")

    except WebSocketDisconnect:
        # Client disconnected during stream
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        await websocket.close(code=1011, reason=str(e))


@router.post(
    "/verify-external",
    status_code=202,
    summary="Verify an external response without generating a new one",
)
async def verify_external_message(
    chat_id: UUID,
    body: MessageCreateRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Submit an externally generated LLM response (or article) to be verified
    by the HalluciSense pipeline. Returns the ID of the created message.
    """
    service = MessageService(db)
    msg_id = await service.verify_external_response(chat_id, current_user.id, body.content)
    return {"message_id": str(msg_id)}
