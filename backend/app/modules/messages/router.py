"""
Messages router — HTTP interface for Chat Messages and WebSockets.
"""
import json
import traceback
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.database.session import AsyncSessionLocal, get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.messages.schemas import (
    MessageCreateRequest,
    MessageListResponse,
    MessageResponse,
)
from app.modules.messages.service import MessageService
from app.core.security import decode_token

logger = structlog.get_logger(__name__)

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
    messages = await service.get_history(chat_id, current_user.id)
    return MessageListResponse(
        items=[MessageResponse.model_validate(m) for m in messages],
        total=len(messages),
    )


@router.websocket("/stream")
async def stream_chat_messages(
    websocket: WebSocket,
    chat_id: UUID,
    token: str = Query(..., description="JWT Bearer token for authentication"),
):
    """
    WebSocket endpoint for streaming LLM generation and running the Verification Pipeline.
    Expects client to send JSON payload: {"content": "User prompt text", "model": "gemini-1.5-flash"}
    """
    await websocket.accept()
    logger.info("websocket_connection_accepted", chat_id=str(chat_id), token_len=len(token))

    try:
        # Validate WebSocket token manually
        try:
            payload = decode_token(token)
            user_id_str = payload.get("sub")
            if not user_id_str:
                logger.warning("websocket_jwt_missing_sub", payload=payload)
                await websocket.close(code=1008, reason="Invalid token payload")
                return
            user_id = UUID(user_id_str)
            logger.info("websocket_auth_success", user_id=str(user_id), chat_id=str(chat_id))
        except Exception as e:
            logger.error("websocket_auth_failed", token_len=len(token), error=str(e), traceback=traceback.format_exc())
            await websocket.close(code=1008, reason=f"Token invalid or expired: {str(e)}")
            return

        # Wait for user input
        data = await websocket.receive_text()
        logger.info("websocket_received_payload", chat_id=str(chat_id), raw_data=data[:200])

        try:
            parsed_data = json.loads(data)
        except Exception as json_err:
            logger.error("websocket_json_parse_error", error=str(json_err))
            await websocket.close(code=1003, reason="Invalid JSON payload")
            return

        user_content = parsed_data.get("content")
        selected_model = parsed_data.get("model") or parsed_data.get("model_slug")

        logger.info(
            "websocket_parsed_payload",
            chat_id=str(chat_id),
            user_id=str(user_id),
            selected_model=selected_model,
            content_length=len(user_content) if user_content else 0,
        )

        if not user_content or not str(user_content).strip():
            logger.warning("websocket_missing_content", parsed_data=parsed_data)
            await websocket.close(code=1003, reason="Content missing")
            return

        async with AsyncSessionLocal() as session:
            service = MessageService(session)

            logger.info("websocket_starting_stream_reply", chat_id=str(chat_id), model=selected_model)
            async for chunk in service.stream_reply(
                chat_id=chat_id,
                user_id=user_id,
                user_content=user_content,
                model_slug=selected_model,
            ):
                await websocket.send_json(chunk)

            logger.info("websocket_stream_completed_normally", chat_id=str(chat_id))

            # Close normally after verification completes
            try:
                await websocket.close(code=1000, reason="Completed")
            except Exception:
                pass

    except WebSocketDisconnect:
        logger.info("websocket_disconnected_by_client", chat_id=str(chat_id))
    except Exception as e:
        logger.error(
            "websocket_stream_uncaught_exception",
            chat_id=str(chat_id),
            error=str(e),
            traceback=traceback.format_exc(),
        )
        try:
            await websocket.send_json({"type": "error", "message": f"Provider Error: {str(e)}"})
            await websocket.close(code=1000, reason="Provider Error")
        except Exception:
            pass


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
