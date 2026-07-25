"""
Celery Tasks for Verification Pipeline.
"""
import asyncio
from typing import List, Optional
from uuid import UUID

from celery import shared_task
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import AsyncSessionLocal
from app.models.message import Message
from app.models.verification_report import VerificationReport
from app.models.sentence_analysis import SentenceAnalysis
from app.models.evidence_item import EvidenceItem
from app.core.constants import VerificationStatus
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.modules.knowledge.retriever import HybridRetriever

async def run_verification_async(message_id: UUID, full_ai_text: str, token_probs: List[float]):
    """Async core of the verification task to interface with DB and async pipeline."""
    pipeline = HallucinationDetectionPipeline()
    retriever = HybridRetriever()
    
    async with AsyncSessionLocal() as session:
        # Step 1: Retrieve necessary facts to pass as evidence items
        # In a real app we'd extract claims first. For Sprint 3, we'll split by sentence and use it as a claim
        # Since pipeline expects EvidenceItems, we can fetch them using HybridRetriever and convert
        claims = [full_ai_text] # simplify for now, ideally extract claims via an LLM call first
        
        raw_evidence = retriever.retrieve(claims)
        
        # Convert to pipeline's expected format (which currently might just take dicts or EvidenceItem objects)
        from app.core.engine.types import EvidenceItem as PipelineEvidenceItem
        
        evidence_items = []
        for e in raw_evidence:
            evidence_items.append(PipelineEvidenceItem(
                claim=claims[0],
                snippet=e["snippet"],
                source_name=e["source_name"],
                source_url=e.get("source_url", ""),
                similarity_score=e["similarity_score"],
                is_supporting=e.get("is_supporting", True)
            ))
            
        # Step 2: Run pipeline
        result = pipeline.analyze_response(
            full_text=full_ai_text, 
            token_probabilities=token_probs,
            evidence_items=evidence_items
        )
        
        # Step 3: Fetch message and update
        # Using SQLAlchemy async session
        from sqlalchemy import select
        msg = await session.get(Message, message_id)
        if not msg:
            return
            
        # Create report and persist
        report = VerificationReport(
            message_id=message_id,
            overall_h_score=result.overall_h_score,
            overall_risk_level=result.overall_risk_level.value,
            factual_error_score=result.pillar1_summary.average_similarity,
            confidence_gap_score=result.pillar2_summary.aggregate_entropy,
            consistency_failure_score=result.pillar3_summary.contradiction_score,
            weights_used=result.weights_used,
            processing_time_ms=0.0 # Time isn't measured inside async block purely yet
        )
        session.add(report)
        await session.flush()
        
        for idx, sentence_res in enumerate(result.sentence_analyses):
            sa = SentenceAnalysis(
                report_id=report.id,
                sentence_index=idx,
                sentence_text=sentence_res.text,
                h_score=sentence_res.hallucination_score,
                risk_level=sentence_res.risk_level.value,
                color_code=sentence_res.color_code,
                factual_error=sentence_res.factual_error,
                confidence_gap=sentence_res.confidence_gap,
                consistency_failure=sentence_res.consistency_failure
            )
            session.add(sa)
            await session.flush()

            for ev in sentence_res.evidence:
                ev_item = EvidenceItem(
                    sentence_analysis_id=sa.id,
                    claim=ev.claim,
                    snippet=ev.snippet,
                    source_name=ev.source_name,
                    source_url=ev.source_url,
                    similarity_score=ev.similarity_score,
                    is_supporting=ev.is_supporting
                )
                session.add(ev_item)
                
        msg.verification_status = VerificationStatus.COMPLETE
        await session.commit()
        # Emit a websocket event to notify client via Redis PubSub
        from app.core.pubsub import publish_message
        await publish_message(f"user_{msg.user_id}", {
            "type": "verification_complete",
            "message_id": str(msg.id),
            "report_id": str(report.id),
            "chat_id": str(msg.chat_id)
        })

@shared_task(name="verify_response")
def verify_response_task(message_id_str: str, full_ai_text: str, token_probs: List[float]):
    """
    Celery task that orchestrates the verification pipeline.
    Runs asynchronously and updates PostgreSQL on completion.
    """
    message_id = UUID(message_id_str)
    
    # Run the async core in the current event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_verification_async(message_id, full_ai_text, token_probs))
    
    return {"status": "success", "message_id": message_id_str}
