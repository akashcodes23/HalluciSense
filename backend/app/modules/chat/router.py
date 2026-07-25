"""
Chat router — HTTP interface for Chat CRUD.
"""
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.modules.auth.dependencies import CurrentUser
from app.modules.chat.schemas import (
    ChatCreateRequest,
    ChatListResponse,
    ChatResponse,
    ChatUpdateRequest,
)
from app.modules.chat.service import ChatService

router = APIRouter(prefix="/chats", tags=["Chats"])


@router.get(
    "",
    response_model=ChatListResponse,
    summary="List all active chats for the authenticated user",
)
async def list_chats(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ChatListResponse:
    service = ChatService(db)
    chats, total = await service.get_user_chats(
        user_id=current_user.id, limit=limit, offset=offset
    )
    return ChatListResponse(
        items=[ChatResponse.model_validate(chat) for chat in chats],
        total=total,
    )


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
)
async def create_chat(
    body: ChatCreateRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChatResponse:
    service = ChatService(db)
    chat = await service.create_chat(
        user_id=current_user.id,
        title=body.title or "New Chat",
        model_used=body.model_used or current_user.preferred_model,
    )
    return ChatResponse.model_validate(chat)


@router.get(
    "/{chat_id}",
    response_model=ChatResponse,
    summary="Get a specific chat by ID",
)
async def get_chat(
    chat_id: UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChatResponse:
    service = ChatService(db)
    chat = await service.get_chat(chat_id=chat_id, user_id=current_user.id)
    return ChatResponse.model_validate(chat)


@router.patch(
    "/{chat_id}",
    response_model=ChatResponse,
    summary="Update a chat (e.g. rename or archive)",
)
async def update_chat(
    chat_id: UUID,
    body: ChatUpdateRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChatResponse:
    service = ChatService(db)
    chat = await service.update_chat(
        chat_id=chat_id,
        user_id=current_user.id,
        title=body.title,
        is_archived=body.is_archived,
    )
    return ChatResponse.model_validate(chat)


@router.delete(
    "/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat permanently",
)
async def delete_chat(
    chat_id: UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    service = ChatService(db)
    await service.delete_chat(chat_id=chat_id, user_id=current_user.id)
