"""
MessageService — use cases for handling messages, LLM generation, and Verification Pipeline integration.
"""
import time
from typing import AsyncGenerator, List, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from starlette.concurrency import run_in_threadpool

from app.core.constants import MessageRole, VerificationStatus
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.exceptions import NotFoundError
from app.workers.celery_app import celery_app
from app.workers.tasks.verification_task import verify_response_task
from app.models.chat import Chat
from app.models.message import Message
from app.models.verification_report import VerificationReport
from app.models.sentence_analysis import SentenceAnalysis
from app.models.evidence_item import EvidenceItem
from app.modules.chat.service import ChatService
from app.modules.orchestrator.service import LLMOrchestrator
from app.repositories.message_repository import MessageRepository


class MessageService:
    """Handles messaging and real-time generation."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = MessageRepository(session)
        self._chat_service = ChatService(session)

    async def get_chat_messages(self, chat_id: UUID, user_id: UUID) -> Tuple[List[Message], int]:
        """Fetch all messages for a chat, ensuring user owns the chat."""
        await self._chat_service.get_chat(chat_id, user_id)
        messages = await self._repo.get_messages_by_chat_id(chat_id)
        return messages, len(messages)

    async def stream_reply(self, chat_id: UUID, user_id: UUID, user_content: str) -> AsyncGenerator[dict, None]:
        """
        1. Save user message.
        2. Fetch chat history.
        3. Stream LLM response.
        4. Save AI message.
        5. Run HalluciSense verification pipeline in background/synchronously (for Sprint 2, synchronous before final WS close).
        """
        # Ensure chat belongs to user and fetch chat model configuration
        chat = await self._chat_service.get_chat(chat_id, user_id)
        
        # Save User Message
        user_msg = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.USER,
            content=user_content,
            verification_status=VerificationStatus.COMPLETE  # User messages don't need verification
        )
        await self._repo.create(user_msg)

        # Build history for LLM
        history_msgs = await self._repo.get_messages_by_chat_id(chat_id)
        provider_messages = [
            {
                "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                "content": m.content,
            }
            for m in history_msgs
        ]
        
        # Stream from orchestrator
        orchestrator = LLMOrchestrator(primary_model=chat.model_used)
        generator = orchestrator.stream_chat(provider_messages)

        full_ai_text = ""
        captured_logits = []
        start_time = time.perf_counter()

        async for chunk in generator:
            if chunk.text:
                full_ai_text += chunk.text
                yield {"type": "token", "text": chunk.text}
            
            if chunk.logits:
                captured_logits.extend([{"token": l.token, "logprob": l.logprob} for l in chunk.logits])
                
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Save AI Message
        ai_msg = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.ASSISTANT,
            content=full_ai_text,
            raw_logits=captured_logits if captured_logits else None,
            processing_time_ms=processing_time_ms,
            verification_status=VerificationStatus.PROCESSING
        )
        ai_msg = await self._repo.create(ai_msg)
        
        # Update chat last_message_at
        await self._chat_service.update_chat(chat_id, user_id) # Using a separate method for timestamp update is ideal, but passing no args just verifies it.
        chat.last_message_at = ai_msg.created_at
        self._session.add(chat)
        await self._session.flush()

        # Persist messages permanently before starting async verification task
        await self._session.commit()

        # Run Verification Pipeline (Module 1 Integration) asynchronously via Celery
        # Map logits to the format expected by pipeline (list of floats)
        token_probs = []
        import math

        token_probs = None

        if captured_logits:
            token_probs = [
                max(0.0, min(1.0, math.exp(l["logprob"])))
                for l in captured_logits
            ]

        # Dispatch background task without blocking the event loop
        await run_in_threadpool(verify_response_task.delay, str(ai_msg.id), full_ai_text, token_probs)

        # We return the message ID immediately. The client will be notified via WebSocket
        # when the verification task completes and updates the database.
        yield {"type": "verification_dispatched", "message_id": str(ai_msg.id)}

    async def verify_external_response(self, chat_id: UUID, user_id: UUID, content: str) -> UUID:
        """
        Takes an externally generated response (pasted by user), saves it as an ASSISTANT message,
        and triggers the verification pipeline without generating a new response.
        """
        # Ensure chat belongs to user
        chat = await self._chat_service.get_chat(chat_id, user_id)
        
        # Save AI Message
        ai_msg = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=MessageRole.ASSISTANT,
            content=content,
            raw_logits=None,
            processing_time_ms=0,
            verification_status=VerificationStatus.PROCESSING
        )
        ai_msg = await self._repo.create(ai_msg)
        
        # Update chat last_message_at
        await self._chat_service.update_chat(chat_id, user_id)
        chat.last_message_at = ai_msg.created_at
        self._session.add(chat)
        await self._session.flush()
        
        # Persist message permanently before async verification task
        await self._session.commit()

        # Dispatch background task — None signals P2 unavailable (no logprobs)
        await run_in_threadpool(verify_response_task.delay, str(ai_msg.id), content, None)

        return ai_msg.id
