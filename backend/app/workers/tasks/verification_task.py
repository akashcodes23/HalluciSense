"""
Celery task for the HalluciSense verification pipeline.

This module receives an AI-generated response, runs the
HalluciSense verification pipeline, stores the verification
results in the database, and notifies the frontend when the
analysis is complete.
"""

import asyncio
import time
from typing import List, Optional
from uuid import UUID

import structlog
from celery import shared_task

from app.database.session import AsyncSessionLocal
from app.models.chat import Chat
from app.models.message import Message
from app.models.verification_report import VerificationReport
from app.models.sentence_analysis import SentenceAnalysis
from app.models.evidence_item import EvidenceItem

from app.core.constants import VerificationStatus
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.modules.knowledge.retriever import HybridRetriever
from app.modules.orchestrator.service import LLMOrchestrator
from sqlalchemy import select, delete


logger = structlog.get_logger(__name__)


def _pillar_to_dict(pillar_result) -> Optional[dict]:
    """Safely serialize a pillar result to a JSON-compatible dict."""
    if pillar_result is None:
        return None
    try:
        return pillar_result.model_dump()
    except AttributeError:
        pass
    try:
        import dataclasses
        return dataclasses.asdict(pillar_result)
    except Exception:
        return None


async def run_verification_async(
    message_id: UUID,
    full_ai_text: str,
    token_probs: Optional[List[float]],
):
    """
    Execute the HalluciSense verification pipeline asynchronously.
    """

    logger.info(
        "verification_task_started",
        message_id=str(message_id),
    )

    start_time = time.perf_counter()

    pipeline = HallucinationDetectionPipeline()
    retriever = HybridRetriever()

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import NullPool
    from app.core.config import settings

    worker_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    WorkerSessionLocal = async_sessionmaker(worker_engine, expire_on_commit=False)

    try:
        async with WorkerSessionLocal() as session:
            # ---------------------------------------------------------
            # STEP 1 — Split AI response into claims
            # ---------------------------------------------------------

            sentences = pipeline._split_sentences(full_ai_text)

            if sentences:
                claims = [
                    sentence[0]
                    if isinstance(sentence, (tuple, list))
                    else sentence
                    for sentence in sentences
                ]
            else:
                claims = [full_ai_text]

        # ---------------------------------------------------------
        # STEP 2 — Retrieve external evidence
        # ---------------------------------------------------------

        logger.info(
            "evidence_retrieval_started",
            message_id=str(message_id),
            num_claims=len(claims),
        )

        retrieval_start = time.perf_counter()

        raw_evidence = retriever.retrieve(claims)

        # Handle async retrievers if necessary
        if asyncio.iscoroutine(raw_evidence):
            raw_evidence = await raw_evidence

        raw_evidence = raw_evidence or []

        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        logger.info(
            "evidence_retrieval_completed",
            message_id=str(message_id),
            evidence_count=len(raw_evidence),
            retrieval_time_ms=round(retrieval_time_ms, 1),
        )

        # ---------------------------------------------------------
        # STEP 3 — Convert evidence into pipeline format
        # ---------------------------------------------------------

        from app.core.engine.types import (
            EvidenceItem as PipelineEvidenceItem,
        )

        evidence_items = []

        for evidence in raw_evidence:

            if isinstance(evidence, dict):

                evidence_items.append(
                    PipelineEvidenceItem(
                        claim=evidence.get(
                            "claim",
                            claims[0],
                        ),
                        snippet=evidence.get(
                            "snippet",
                            "",
                        ),
                        source_name=evidence.get(
                            "source_name",
                            "Unknown",
                        ),
                        source_url=evidence.get(
                            "source_url",
                            "",
                        ),
                        similarity_score=evidence.get(
                            "similarity_score",
                            0.0,
                        ),
                        is_supporting=evidence.get(
                            "is_supporting",
                            True,
                        ),
                    )
                )

            else:

                evidence_items.append(
                    PipelineEvidenceItem(
                        claim=getattr(
                            evidence,
                            "claim",
                            claims[0],
                        ),
                        snippet=getattr(
                            evidence,
                            "snippet",
                            "",
                        ),
                        source_name=getattr(
                            evidence,
                            "source_name",
                            "Unknown",
                        ),
                        source_url=getattr(
                            evidence,
                            "source_url",
                            "",
                        ),
                        similarity_score=getattr(
                            evidence,
                            "similarity_score",
                            0.0,
                        ),
                        is_supporting=getattr(
                            evidence,
                            "is_supporting",
                            True,
                        ),
                    )
                )

        # ---------------------------------------------------------
        # STEP 3.5 — Generate Alternate Samples for Pillar 3
        # ---------------------------------------------------------

        sample_responses: List[str] = []

        logger.info(
            "pillar3_sample_generation_started",
            message_id=str(message_id),
            requested_samples=3,
        )

        sample_start = time.perf_counter()

        try:
            target_msg = await session.get(Message, message_id)

            if target_msg is not None:
                target_chat = await session.get(Chat, target_msg.chat_id)

                if target_chat is not None:
                    history_query = (
                        select(Message)
                        .where(Message.chat_id == target_msg.chat_id)
                        .where(Message.created_at <= target_msg.created_at)
                        .order_by(Message.created_at.asc())
                    )
                    history_res = await session.execute(history_query)
                    all_chat_msgs = history_res.scalars().all()

                    prompt_messages = [
                        {
                            "role": (
                                msg.role.value
                                if hasattr(msg.role, "value")
                                else str(msg.role)
                            ),
                            "content": msg.content,
                        }
                        for msg in all_chat_msgs
                        if msg.id != message_id
                    ]

                    if prompt_messages:
                        orchestrator = LLMOrchestrator(
                            primary_model=target_chat.model_used
                        )

                        sample_responses = await orchestrator.generate_samples(
                            messages=prompt_messages,
                            count=3,
                            temperature=0.7,
                        )

            sample_time_ms = (time.perf_counter() - sample_start) * 1000

            logger.info(
                "pillar3_sample_generation_completed",
                message_id=str(message_id),
                requested_samples=3,
                generated_samples=len(sample_responses),
                sample_time_ms=round(sample_time_ms, 1),
            )

        except Exception as sample_exc:
            sample_time_ms = (time.perf_counter() - sample_start) * 1000

            logger.warning(
                "pillar3_sample_generation_failed",
                message_id=str(message_id),
                requested_samples=3,
                generated_samples=0,
                error=str(sample_exc),
                sample_time_ms=round(sample_time_ms, 1),
            )
            sample_responses = []

        # ---------------------------------------------------------
        # STEP 4 — Run HalluciSense hybrid pipeline
        # ---------------------------------------------------------

        logger.info(
            "pipeline_analysis_started",
            message_id=str(message_id),
            num_samples=len(sample_responses),
            token_probs_available=token_probs is not None,
        )

        pipeline_start = time.perf_counter()

        result = pipeline.analyze_response(
            full_text=full_ai_text,
            token_probabilities=token_probs,
            evidence_items=evidence_items,
            sample_responses=sample_responses,
        )

        # Support async pipeline implementation
        if asyncio.iscoroutine(result):
            result = await result

        pipeline_time_ms = (time.perf_counter() - pipeline_start) * 1000

        # ---------------------------------------------------------
        # STEP 5 — Retrieve original message
        # ---------------------------------------------------------

        msg = await session.get(
            Message,
            message_id,
        )

        if msg is None:

            logger.error(
                "verification_message_not_found",
                message_id=str(message_id),
            )

            return {
                "status": "error",
                "reason": "message_not_found",
            }

        processing_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "pipeline_analysis_completed",
            message_id=str(message_id),
            overall_h_score=result.overall_h_score,
            risk_level=(
                result.overall_risk_level.value
                if hasattr(result.overall_risk_level, "value")
                else str(result.overall_risk_level)
            ),
            pipeline_time_ms=round(pipeline_time_ms, 1),
            processing_time_ms=round(processing_time_ms, 1),
            pillar1_available=True,
            pillar2_available=getattr(result.pillar2_summary, "available", False),
            pillar3_available=getattr(result.pillar3_summary, "available", False),
        )

        # ---------------------------------------------------------
        # STEP 5.5 — Idempotency: remove existing report if present
        # ---------------------------------------------------------

        existing_report = await session.execute(
            select(VerificationReport).where(
                VerificationReport.message_id == message_id
            )
        )
        existing = existing_report.scalar_one_or_none()

        if existing is not None:
            logger.info(
                "verification_replacing_existing_report",
                message_id=str(message_id),
                existing_report_id=str(existing.id),
            )
            # Cascade deletes SentenceAnalysis and EvidenceItem rows
            await session.delete(existing)
            await session.flush()

        # ---------------------------------------------------------
        # STEP 6 — Save overall verification report
        # ---------------------------------------------------------

        logger.info(
            "verification_persistence_started",
            message_id=str(message_id),
        )

        report = VerificationReport(
            message_id=message_id,

            overall_h_score=result.overall_h_score,

            overall_risk_level=(
                result.overall_risk_level.value
                if hasattr(
                    result.overall_risk_level,
                    "value",
                )
                else str(result.overall_risk_level)
            ),

            factual_error_score=(
                result.pillar1_summary.factual_error_score
                if result.pillar1_summary is not None
                else None
            ),

            confidence_gap_score=(
                result.pillar2_summary.confidence_gap_score
                if result.pillar2_summary is not None
                else None
            ),

            consistency_failure_score=(
                result.pillar3_summary.consistency_failure_score
                if result.pillar3_summary is not None
                else None
            ),

            weights_used=result.weights_used,

            pillar1_summary=_pillar_to_dict(result.pillar1_summary),
            pillar2_summary=_pillar_to_dict(result.pillar2_summary),
            pillar3_summary=_pillar_to_dict(result.pillar3_summary),

            corrected_response=result.corrected_response,

            processing_time_ms=processing_time_ms,
        )

        session.add(report)

        await session.flush()

        # ---------------------------------------------------------
        # STEP 7 — Save sentence-level analyses
        # ---------------------------------------------------------

        for index, sentence_result in enumerate(
            result.sentence_analyses
        ):

            risk_level = sentence_result.risk_level

            if hasattr(risk_level, "value"):
                risk_level = risk_level.value

            sentence_analysis = SentenceAnalysis(
                report_id=report.id,

                sentence_index=index,

                sentence_text=sentence_result.text,

                start_char=sentence_result.start_char,

                end_char=sentence_result.end_char,

                h_score=(
                    sentence_result.hallucination_score
                ),

                risk_level=str(risk_level),

                color_code=sentence_result.color_code,

                factual_error=sentence_result.factual_error,

                confidence_gap=sentence_result.confidence_gap,

                consistency_failure=(
                    sentence_result.consistency_failure
                ),

                reasoning=sentence_result.reasoning,
            )

            session.add(sentence_analysis)

            await session.flush()

            # -----------------------------------------------------
            # STEP 8 — Save evidence for this sentence
            # -----------------------------------------------------

            sentence_evidence = (
                sentence_result.evidence or []
            )

            for evidence in sentence_evidence:

                evidence_db = EvidenceItem(
                    sentence_analysis_id=(
                        sentence_analysis.id
                    ),

                    claim=evidence.claim,

                    snippet=evidence.snippet,

                    source_name=evidence.source_name,

                    source_url=evidence.source_url,

                    similarity_score=(
                        evidence.similarity_score
                    ),

                    is_supporting=(
                        evidence.is_supporting
                    ),
                )

                session.add(evidence_db)

        # ---------------------------------------------------------
        # STEP 9 — Mark verification complete
        # ---------------------------------------------------------

        msg.verification_status = (
            VerificationStatus.COMPLETE
        )

        await session.commit()

        logger.info(
            "verification_persistence_completed",
            message_id=str(message_id),
            report_id=str(report.id),
            sentence_count=len(result.sentence_analyses),
        )

        # ---------------------------------------------------------
        # STEP 10 — Notify frontend
        # ---------------------------------------------------------

        try:

            from app.core.pubsub import publish_message

            await publish_message(
                f"user_{msg.user_id}",
                {
                    "type": "verification_complete",
                    "message_id": str(msg.id),
                    "report_id": str(report.id),
                    "chat_id": str(msg.chat_id),
                },
            )

        except Exception as exc:

            # Verification itself has succeeded.
            # A notification failure should not destroy the report.

            logger.exception(
                "verification_pubsub_failed",
                message_id=str(message_id),
                error=str(exc),
            )

        logger.info(
            "verification_task_completed",
            message_id=str(message_id),
            report_id=str(report.id),
            overall_h_score=result.overall_h_score,
            risk_level=(
                result.overall_risk_level.value
                if hasattr(result.overall_risk_level, "value")
                else str(result.overall_risk_level)
            ),
            processing_time_ms=round(processing_time_ms, 1),
        )

        return {
            "status": "success",
            "message_id": str(message_id),
            "report_id": str(report.id),
            "overall_h_score": result.overall_h_score,
        }
    finally:
        await worker_engine.dispose()


# =============================================================
# CELERY ENTRY POINT
# =============================================================

@shared_task(
    name="verify_response",
    bind=True,
)
def verify_response_task(
    self,
    message_id_str: str,
    full_ai_text: str,
    token_probs: Optional[List[float]] = None,
):
    """
    Celery entry point for HalluciSense verification.
    """

    logger.info(
        "celery_verify_response_received",
        message_id=message_id_str,
    )

    try:

        result = asyncio.run(
            run_verification_async(
                UUID(message_id_str),
                full_ai_text,
                token_probs,
            )
        )

        return result

    except Exception as exc:

        logger.exception(
            "verification_task_failed",
            message_id=message_id_str,
            error=str(exc),
            error_type=type(exc).__name__,
        )

        raise