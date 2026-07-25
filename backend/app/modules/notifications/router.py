"""
Notifications Router.
Provides a global WebSocket endpoint for real-time events.
"""
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.pubsub import subscribe
from app.core.security import decode_token

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/notifications", tags=["Notifications"])

@router.websocket("/ws")
async def notifications_websocket(
    websocket: WebSocket,
    token: str = Query(..., description="JWT Bearer token for authentication")
):
    """
    Global WebSocket for user notifications (e.g. verification complete).
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
        channel = f"user_{user_id}"
        
        # Subscribe to user's personal channel in Redis
        async for message in subscribe(channel):
            await websocket.send_json(message)
            
    except WebSocketDisconnect:
        logger.info("websocket_disconnected", user_id=str(user_id_str) if 'user_id_str' in locals() else "unknown")
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        await websocket.close(code=1011, reason="Internal error")
