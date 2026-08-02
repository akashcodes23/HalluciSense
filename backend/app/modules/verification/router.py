import time
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.modules.knowledge.retriever import HybridRetriever
from app.core.engine.types import EvidenceItem

router = APIRouter(prefix="/verification", tags=["Verification Engine"])

class VerifyRequest(BaseModel):
    text: str

@router.post("/verify-text")
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

    # Ensure Pydantic model dump works
    try:
        sentences_out = [s.model_dump() for s in report.sentence_analyses]
    except AttributeError:
        # Fallback if using dataclasses
        import dataclasses
        sentences_out = [dataclasses.asdict(s) for s in report.sentence_analyses]

    # Safely compute confidence_score — NEVER do 1.0 - None
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
