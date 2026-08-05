"""
Verification Router — HTTP Interface for HalluciSense Verification Pipeline endpoints.
"""
import time
from uuid import UUID
from typing import Annotated, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.modules.auth.dependencies import get_current_user
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.modules.knowledge.retriever import HybridRetriever
from app.core.engine.types import EvidenceItem
from app.repositories.verification_repository import VerificationRepository
from app.modules.messages.schemas import VerificationReportResponse

router = APIRouter(prefix="/verification", tags=["Verification Engine"])


class VerifyRequest(BaseModel):
    text: str


@router.get("/{message_id}", response_model=VerificationReportResponse, summary="Get Verification Report by Message ID")
async def get_verification_report(
    message_id: UUID,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VerificationReportResponse:
    """Fetch complete verification report for a specific message."""
    repo = VerificationRepository(db)
    report = await repo.get_report_by_message_id(message_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification report for message {message_id} not found."
        )
    return VerificationReportResponse.model_validate(report)


@router.get("/{message_id}/sentence/{sentence_index}", summary="Get Specific Sentence Analysis Detail")
async def get_sentence_detail(
    message_id: UUID,
    sentence_index: int,
    current_user: Annotated[dict, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Fetch sentence-level analysis detail by sentence index."""
    repo = VerificationRepository(db)
    report = await repo.get_report_by_message_id(message_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification report for message {message_id} not found."
        )
    
    matching = [s for s in report.sentence_analyses if s.sentence_index == sentence_index]
    if not matching:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sentence index {sentence_index} not found in report."
        )
    return matching[0]


@router.post("/verify-text", summary="Direct Text Verification")
def verify_text(request: VerifyRequest):
    start_time = time.perf_counter()
    pipeline = HallucinationDetectionPipeline()
    retriever = HybridRetriever()

    text = request.text.strip()
    if not text:
        return {"error": "Text cannot be empty."}

    # Extract sentences as claims
    sentences = pipeline._split_sentences(text)
    claims = [s[0] for s in sentences] if sentences else [text]

    # Retrieve evidence
    raw_evidence = retriever.retrieve(claims)
    
    # Map to Pipeline EvidenceItems
    evidence_items = []
    for e in raw_evidence:
        evidence_items.append(EvidenceItem(
            claim=claims[0],
            snippet=e["snippet"],
            source_name=e["source_name"],
            source_url=e.get("source_url", ""),
            similarity_score=e.get("similarity_score", 0.9),
            is_supporting=e.get("is_supporting", True)
        ))

    # Run verification pipeline
    report = pipeline.analyze_response(
        full_text=text,
        evidence_items=evidence_items
    )

    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000

    try:
        sentences_out = [s.model_dump() for s in report.sentence_analyses]
    except AttributeError:
        import dataclasses
        sentences_out = [dataclasses.asdict(s) for s in report.sentence_analyses]

    cg_score = report.pillar2_summary.confidence_gap_score
    confidence_score = round(1.0 - cg_score, 4) if cg_score is not None else None

    return {
        "overall_h_score": report.overall_h_score,
        "risk_level": report.overall_risk_level.value if hasattr(report.overall_risk_level, 'value') else report.overall_risk_level,
        "trust_score": round(1.0 - report.overall_h_score, 4),
        "confidence_score": confidence_score,
        "evidence_coverage": round(1.0 - report.pillar1_summary.factual_error_score, 4),

        "pillars": {
            "factual_error": report.pillar1_summary.factual_error_score,
            "confidence_gap": report.pillar2_summary.confidence_gap_score,
            "consistency_failure": report.pillar3_summary.consistency_failure_score,
        },

        "pillar_availability": {
            "pillar1": True,
            "pillar2": getattr(report.pillar2_summary, "available", False),
            "pillar3": getattr(report.pillar3_summary, "available", False),
        },

        "weights_used": report.weights_used,

        "verified_claims": len([s for s in report.sentence_analyses if (s.risk_level.value if hasattr(s.risk_level, 'value') else s.risk_level) == "VERIFIED"]),
        "hallucinated_claims": len([s for s in report.sentence_analyses if (s.risk_level.value if hasattr(s.risk_level, 'value') else s.risk_level) != "VERIFIED"]),
        "sentence_analysis": sentences_out,
        "corrected_response": report.corrected_response,
        "summary": "Verification completed successfully.",
        "processing_time": processing_time_ms
    }
