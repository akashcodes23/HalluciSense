"""
Message Service.
Handles chat message persistence, history retrieval, LLM streaming, and verification task dispatch.
"""
import asyncio
import math
import time
import traceback
from typing import AsyncGenerator, List, Optional
from uuid import UUID

import structlog
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import MessageRole, VerificationStatus
from app.models.message import Message
from app.modules.chat.service import ChatService
from app.modules.orchestrator.service import LLMOrchestrator
from app.repositories.message_repository import MessageRepository
from app.workers.tasks.verification_task import verify_response_task, run_verification_async

logger = structlog.get_logger(__name__)


class MessageService:
    def __init__(
        self,
        session: AsyncSession,
        message_repo: Optional[MessageRepository] = None,
        chat_service: Optional[ChatService] = None,
    ):
        self._session = session
        self._repo = message_repo or MessageRepository(session)
        self._chat_service = chat_service or ChatService(session)

    async def get_history(
        self, chat_id: UUID, user_id: UUID, limit: int = 50, offset: int = 0
    ) -> List[Message]:
        """Fetch chat history, verifying ownership."""
        await self._chat_service.get_chat(chat_id, user_id)
        return await self._repo.get_messages_by_chat_id(
            chat_id, limit=limit, offset=offset
        )

    async def stream_reply(
        self,
        chat_id: UUID,
        user_id: UUID,
        user_content: str,
        model_slug: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream an LLM reply for a chat message and trigger async verification.
        """
        # Validate chat ownership
        chat = await self._chat_service.get_chat(chat_id, user_id)

        # Active model precedence: explicit model > chat stored model > system default
        active_model = model_slug or chat.model_used or "gemini-1.5-flash"

        # Update chat model_used if user selected a different model from dropdown
        if model_slug and chat.model_used != model_slug:
            chat.model_used = model_slug
            self._session.add(chat)
            await self._session.flush()

        logger.info(
            "stream_reply_started",
            chat_id=str(chat_id),
            user_id=str(user_id),
            active_model=active_model,
            content_len=len(user_content),
        )

        # 1. Save User Message
        user_msg = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.USER,
            content=user_content,
            verification_status=VerificationStatus.COMPLETE,
        )
        await self._repo.create(user_msg)

        # 2. Build history for orchestrator
        history_msgs = await self._repo.get_messages_by_chat_id(chat_id, limit=20)
        provider_messages = [
            {
                "role": (
                    m.role.value
                    if hasattr(m.role, "value")
                    else str(m.role).lower()
                ),
                "content": m.content,
            }
            for m in history_msgs
            if m.content and m.content.strip()
        ]

        logger.info(
            "stream_reply_history_built",
            history_count=len(provider_messages),
            active_model=active_model,
        )

        # Stream from orchestrator
        orchestrator = LLMOrchestrator(primary_model=active_model)
        generator = orchestrator.stream_chat(provider_messages)

        full_ai_text = ""
        captured_logits = []
        start_time = time.perf_counter()
        token_count = 0

        try:
            async for chunk in generator:
                if chunk.text:
                    full_ai_text += chunk.text
                    token_count += 1
                    yield {"type": "token", "text": chunk.text}

                if chunk.logits:
                    captured_logits.extend(
                        [{"token": l.token, "logprob": l.logprob} for l in chunk.logits]
                    )
        except Exception as e:
            logger.error(
                "message_service_stream_failed",
                chat_id=str(chat_id),
                model=active_model,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            raise e

        processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "stream_reply_streaming_finished",
            chat_id=str(chat_id),
            tokens_count=token_count,
            ai_text_len=len(full_ai_text),
            time_ms=processing_time_ms,
        )

        logger.info(
    "assistant_response_completed",
    model=active_model,
    length=len(full_ai_text),
    response=full_ai_text,
)
        # Validate non-empty response before persistence
        if not full_ai_text or not full_ai_text.strip():
            logger.error(
                "stream_reply_empty_ai_response_error",
                chat_id=str(chat_id),
                model=active_model,
            )
            raise ValueError(f"Provider '{active_model}' returned empty response content.")

        # Save AI Message
        ai_msg = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.ASSISTANT,
            content=full_ai_text,
            raw_logits=captured_logits if captured_logits else None,
            processing_time_ms=processing_time_ms,
            verification_status=VerificationStatus.PROCESSING,
        )
        ai_msg = await self._repo.create(ai_msg)

        # Update chat last_message_at
        chat.last_message_at = ai_msg.created_at
        self._session.add(chat)
        await self._session.flush()
        await self._session.commit()

        # Safely dispatch background verification task
        token_probs = None
        if captured_logits:
            token_probs = [
                max(0.0, min(1.0, math.exp(l["logprob"]))) for l in captured_logits
            ]

        try:
            logger.info("dispatching_verification_task", message_id=str(ai_msg.id))
            await run_in_threadpool(
                verify_response_task.delay, str(ai_msg.id), full_ai_text, token_probs
            )
        except Exception as celery_err:
            logger.warning(
                "celery_task_dispatch_failed_fallback_to_async_task",
                message_id=str(ai_msg.id),
                error=str(celery_err),
            )
            # Spawns background async verification task directly in FastAPI loop if Celery worker is offline
            asyncio.create_task(
                run_verification_async(ai_msg.id, full_ai_text, token_probs)
            )

        # Signal completion to WebSocket
        yield {"type": "verification_dispatched", "message_id": str(ai_msg.id)}

    async def verify_external_response(
        self, chat_id: UUID, user_id: UUID, content: str
    ) -> UUID:
        """
        Takes an externally generated response (pasted by user), saves it as an ASSISTANT message,
        and triggers the verification pipeline.
        """
        if not content or not content.strip():
            raise ValueError("Verification content cannot be empty.")

        chat = await self._chat_service.get_chat(chat_id, user_id)

        ai_msg = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.ASSISTANT,
            content=content,
            raw_logits=None,
            processing_time_ms=0,
            verification_status=VerificationStatus.PROCESSING,
        )
        ai_msg = await self._repo.create(ai_msg)

        chat.last_message_at = ai_msg.created_at
        self._session.add(chat)
        await self._session.flush()
        await self._session.commit()

        try:
            logger.info("verify_external_response_dispatched", message_id=str(ai_msg.id))
            await run_in_threadpool(
                verify_response_task.delay, str(ai_msg.id), content, None
            )
        except Exception as celery_err:
            logger.warning(
                "celery_verify_external_dispatch_failed_fallback_to_async_task",
                message_id=str(ai_msg.id),
                error=str(celery_err),
            )
            asyncio.create_task(
                run_verification_async(ai_msg.id, content, None)
            )

        return ai_msg.id
