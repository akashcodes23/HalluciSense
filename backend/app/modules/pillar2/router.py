"""
HalluciSense Pillar 2 — Production FastAPI Router
=================================================
Exposes production REST API endpoints for Pillar 2 Evidence-Aware Multi-LLM Verification.
Endpoints:
  - POST /api/v1/pillar2/verify
  - POST /api/v1/pillar2/claims
  - POST /api/v1/pillar2/evidence
  - POST /api/v1/pillar2/consensus
  - POST /api/v1/pillar2/hallucination-score
  - GET  /api/v1/pillar2/providers
  - GET  /api/v1/pillar2/health
  - GET  /api/v1/pillar2/version
"""

import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.pillar2.claim_extraction.extractor import ClaimExtractionEngine
from app.pillar2.claim_extraction.schemas import ClaimExtractionRequest, ClaimExtractionResponse
from app.pillar2.consensus_engine.engine import ConsensusEngine
from app.pillar2.consensus_engine.schemas import ConsensusResult
from app.pillar2.contradiction_analysis.analyzer import ContradictionAnalyzer
from app.pillar2.evidence_retrieval.manager import EvidenceRetrievalManager
from app.pillar2.evidence_retrieval.schemas import RetrievalRequest, RetrievalResponse
from app.pillar2.explainability.engine import PillarTwoExplainabilityEngine
from app.pillar2.feature_generation.generator import EvidenceFeatureGenerator
from app.pillar2.knowledge_graph.builder import EntityRelationGraphBuilder
from app.pillar2.multi_llm_verifier.orchestrator import MultiLLMVerificationOrchestrator
from app.pillar2.multi_llm_verifier.schemas import (
    MultiLLMVerificationRequest,
    MultiLLMVerificationResponse,
    SingleClaimVerification,
)
from app.pillar2.schemas.frontend_contract import (
    ClaimCardUI,
    ConfidenceGaugeUI,
    ConsensusViewUI,
    DashboardOverviewUI,
    EvidenceCardUI,
    GraphEdgeUI,
    GraphNodeUI,
    NetworkGraphUI,
    RiskIndicatorUI,
    TimelineItemUI,
)
from app.pillar2.unified_hscore.calculator import UnifiedHScoreCalculator
from app.pillar2.unified_hscore.schemas import RiskCategory, UnifiedHScoreResult

router = APIRouter(prefix="/pillar2", tags=["Pillar 2 Verification Engine"])

# Initialize Pillar 2 engine instances
claim_extractor = ClaimExtractionEngine()
graph_builder = EntityRelationGraphBuilder()
evidence_manager = EvidenceRetrievalManager()
llm_orchestrator = MultiLLMVerificationOrchestrator()
consensus_engine = ConsensusEngine()
contradiction_analyzer = ContradictionAnalyzer()
feature_generator = EvidenceFeatureGenerator()
hscore_calculator = UnifiedHScoreCalculator()
explainability_engine = PillarTwoExplainabilityEngine()


# ── Request / Response Schemas for API ────────────────────────────────────────

class FullVerificationRequest(BaseModel):
    text: str = Field(..., min_length=1, description="LLM response text to verify")
    pillar1_probability: float = Field(default=0.50, ge=0.0, le=1.0, description="Frozen Pillar 1 statistical probability")
    evidence_providers: Optional[List[str]] = Field(None, description="Target evidence providers")
    llm_verifiers: Optional[List[str]] = Field(None, description="Target LLM verifiers")


class FullVerificationResponse(BaseModel):
    verification_id: str
    text: str
    hallucisense_score: UnifiedHScoreResult
    explanation: Any
    dashboard_ui: DashboardOverviewUI
    execution_time_ms: float


class HScoreCalculationRequest(BaseModel):
    pillar1_probability: float = Field(..., ge=0.0, le=1.0)
    support_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    contradiction_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    authority_score: float = Field(default=0.8, ge=0.0, le=1.0)
    consensus_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    max_contradiction_severity: float = Field(default=0.0, ge=0.0, le=1.0)


# ── API Endpoints ─────────────────────────────────────────────────────────────

@router.get("/health", status_code=status.HTTP_200_OK)
async def get_health() -> Dict[str, Any]:
    """Pillar 2 Service Health Check."""
    return {
        "status": "HEALTHY",
        "module": "Pillar 2 Verification Engine",
        "pillar1_firewall": "ACTIVE (Frozen)",
        "evidence_providers": evidence_manager.list_available_providers(),
        "llm_verifiers": llm_orchestrator.list_available_verifiers(),
    }


@router.get("/version", status_code=status.HTTP_200_OK)
async def get_version() -> Dict[str, str]:
    """Pillar 2 Module Version Metadata."""
    return {
        "module": "HalluciSense Pillar 2",
        "version": "10.0.0",
        "pipeline_phase": "Phase 10",
        "pillar1_status": "LOCKED_AND_FROZEN",
    }


@router.get("/providers", status_code=status.HTTP_200_OK)
async def get_providers() -> Dict[str, Any]:
    """List registered evidence retrieval providers and multi-LLM verifiers."""
    return {
        "evidence_providers": evidence_manager.list_available_providers(),
        "llm_verifiers": llm_orchestrator.list_available_verifiers(),
    }


@router.post("/claims", response_model=ClaimExtractionResponse)
async def extract_claims(request: ClaimExtractionRequest) -> ClaimExtractionResponse:
    """Decompose response text into atomic claims."""
    return claim_extractor.extract_claims(request)


@router.post("/evidence", response_model=RetrievalResponse)
async def retrieve_evidence(request: RetrievalRequest) -> RetrievalResponse:
    """Retrieve evidence across providers for a given query."""
    return await evidence_manager.retrieve_evidence(request)


@router.post("/consensus", response_model=ConsensusResult)
async def compute_consensus(claim_id: str, verifications: List[SingleClaimVerification]) -> ConsensusResult:
    """Compute statistical consensus across multi-LLM verifications."""
    return consensus_engine.compute_consensus(claim_id, verifications)


@router.post("/hallucination-score", response_model=UnifiedHScoreResult)
async def compute_hscore(req: HScoreCalculationRequest) -> UnifiedHScoreResult:
    """Calculate 0-100 Unified HalluciSense Score."""
    from app.pillar2.contradiction_analysis.schemas import ContradictionAnalysisResult, ContradictionGraphVisualization
    from app.pillar2.feature_generation.schemas import PillarTwoFeatures

    p2_feats = PillarTwoFeatures(
        support_ratio=req.support_ratio,
        contradiction_ratio=req.contradiction_ratio,
        authority_score=req.authority_score,
        source_diversity=0.5,
        evidence_coverage=0.8,
        evidence_density=1.0,
        citation_quality=0.8,
        consensus_confidence=req.consensus_confidence,
        recency_score=0.8,
        verification_completeness=0.8,
    )
    cnt_res = ContradictionAnalysisResult(
        contradictions=[],
        contradiction_count=0,
        fabrication_index=0.0,
        max_severity=req.max_contradiction_severity,
        graph_visualization=ContradictionGraphVisualization(
            nodes=[], edges=[], total_contradictions=0, high_severity_count=0
        ),
    )
    return hscore_calculator.calculate_hscore(req.pillar1_probability, p2_feats, cnt_res)


@router.post("/verify", response_model=FullVerificationResponse)
async def full_verification(request: FullVerificationRequest) -> FullVerificationResponse:
    """
    Execute full Pillar 2 evidence-aware verification pipeline.
    Decomposes claims, retrieves evidence, executes multi-LLM verification,
    computes consensus & contradiction graphs, generates 10 evidence features,
    fuses with frozen Pillar 1 score, and formats Dashboard UI contract payload.
    """
    t0 = time.perf_counter()
    v_id = f"verif_{uuid.uuid4().hex[:10]}"
    timeline: List[TimelineItemUI] = []

    # 1. Claim Extraction
    t_step = time.perf_counter()
    claim_resp = claim_extractor.extract_claims(ClaimExtractionRequest(text=request.text))
    claims = claim_resp.extracted_claims
    timeline.append(
        TimelineItemUI(
            step_name="Claim Extraction",
            status="COMPLETE",
            latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2),
            description=f"Extracted {len(claims)} atomic claim(s).",
        )
    )

    # 2. Knowledge Graph Construction
    t_step = time.perf_counter()
    graph = graph_builder.build_graph(claims, graph_id=f"g_{v_id}")
    timeline.append(
        TimelineItemUI(
            step_name="Semantic Graph Construction",
            status="COMPLETE",
            latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2),
            description=f"Built graph with {graph.num_nodes} node(s) and {graph.num_edges} edge(s).",
        )
    )

    # 3. Evidence Retrieval
    t_step = time.perf_counter()
    all_evidence = []
    if claims:
        # Retrieve evidence for first 3 claims
        for claim in claims[:3]:
            ret_resp = await evidence_manager.retrieve_evidence(
                RetrievalRequest(
                    query=claim.claim_text,
                    providers=request.evidence_providers,
                    max_results_per_provider=2,
                )
            )
            all_evidence.extend(ret_resp.items)
    timeline.append(
        TimelineItemUI(
            step_name="Multi-Provider Evidence Retrieval",
            status="COMPLETE",
            latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2),
            description=f"Retrieved {len(all_evidence)} evidence item(s).",
        )
    )

    # 4. Multi-LLM Verification & Consensus
    t_step = time.perf_counter()
    consensus_map: Dict[str, ConsensusResult] = {}
    ev_dicts = [e.model_dump() for e in all_evidence]

    for claim in claims:
        multi_resp = await llm_orchestrator.verify_claim_multi_llm(
            MultiLLMVerificationRequest(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                evidence_snippets=ev_dicts,
                verifiers=request.llm_verifiers,
            )
        )
        c_res = consensus_engine.compute_consensus(claim.claim_id, multi_resp.verifications)
        consensus_map[claim.claim_id] = c_res

    timeline.append(
        TimelineItemUI(
            step_name="Multi-LLM Consensus Verification",
            status="COMPLETE",
            latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2),
            description=f"Executed consensus verification across {len(consensus_map)} claim(s).",
        )
    )

    # 5. Contradiction Analysis
    t_step = time.perf_counter()
    cnt_res = contradiction_analyzer.analyze_contradictions(claims, consensus_map, all_evidence)
    timeline.append(
        TimelineItemUI(
            step_name="Contradiction & Severity Analysis",
            status="COMPLETE",
            latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2),
            description=f"Identified {cnt_res.contradiction_count} contradiction(s) (max severity {cnt_res.max_severity:.2f}).",
        )
    )

    # 6. Feature Generation & Unified H-Score
    t_step = time.perf_counter()
    p2_feats = feature_generator.generate_features(claims, all_evidence, consensus_map)
    hscore_res = hscore_calculator.calculate_hscore(
        pillar1_probability=request.pillar1_probability,
        p2_features=p2_feats,
        contradiction_result=cnt_res,
    )
    timeline.append(
        TimelineItemUI(
            step_name="Unified H-Score Fusion",
            status="COMPLETE",
            latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2),
            description=f"Calculated score {hscore_res.hallucisense_score:.1f}/100 ({hscore_res.risk_category.value}).",
        )
    )

    # 7. Explainability
    t_step = time.perf_counter()
    explanation = explainability_engine.generate_explanation(
        claims, all_evidence, consensus_map, cnt_res, p2_feats, hscore_res
    )
    timeline.append(
        TimelineItemUI(
            step_name="Explainability Report Generation",
            status="COMPLETE",
            latency_ms=round((time.perf_counter() - t_step) * 1000.0, 2),
            description="Generated human-readable audit narrative.",
        )
    )

    # 8. Build Dashboard UI Contract Payload
    color_map = {
        RiskCategory.CRITICAL: "#d7191c",
        RiskCategory.HIGH: "#fdae61",
        RiskCategory.MODERATE: "#fee08b",
        RiskCategory.LOW: "#abdda4",
        RiskCategory.VERY_LOW: "#2b83ba",
    }
    risk_ui = RiskIndicatorUI(
        hallucisense_score=hscore_res.hallucisense_score,
        risk_category=hscore_res.risk_category,
        color_hex=color_map.get(hscore_res.risk_category, "#7f7f7f"),
        badge_label=f"{hscore_res.risk_category.value} RISK",
    )
    gauge_ui = ConfidenceGaugeUI(
        overall_confidence=hscore_res.overall_confidence,
        confidence_level="HIGH" if hscore_res.overall_confidence > 0.8 else "MODERATE",
        pillar1_probability=hscore_res.pillar1_probability,
        evidence_quality_score=p2_feats.authority_score,
    )

    claim_cards = [
        ClaimCardUI(
            claim_id=c.claim_id,
            text=c.claim_text,
            claim_type=c.claim_type.value,
            consensus_label=consensus_map.get(c.claim_id).majority_label.value if consensus_map.get(c.claim_id) else "UNKNOWN",
            confidence_percentage=consensus_map.get(c.claim_id).consensus_confidence * 100.0 if consensus_map.get(c.claim_id) else 50.0,
            supporting_sources_count=len(consensus_map.get(c.claim_id).label_distribution) if consensus_map.get(c.claim_id) else 0,
            contradicting_sources_count=len(consensus_map.get(c.claim_id).disagreeing_verifiers) if consensus_map.get(c.claim_id) else 0,
        )
        for c in claims
    ]

    evidence_cards = [
        EvidenceCardUI(
            evidence_id=e.evidence_id,
            title=e.title,
            provider=e.source,
            url=e.url,
            snippet=e.snippet,
            authority_score=e.authority_score,
            publication_date=e.publication_date,
        )
        for e in all_evidence
    ]

    consensus_views = [
        ConsensusViewUI(
            claim_id=cid,
            majority_label=c_obj.majority_label.value,
            agreement_percentage=c_obj.pairwise_agreement_score * 100.0,
            shannon_entropy=c_obj.shannon_entropy,
            label_breakdown=c_obj.label_distribution,
            dissenting_verifiers=[d.verifier_name for d in c_obj.disagreeing_verifiers],
        )
        for cid, c_obj in consensus_map.items()
    ]

    nodes_ui = [GraphNodeUI(id=n["id"], label=n["label"], type=n["type"]) for n in cnt_res.graph_visualization.nodes]
    edges_ui = [GraphEdgeUI(source=e["source"], target=e["target"], relation=e["relation"]) for e in cnt_res.graph_visualization.edges]

    dashboard_ui = DashboardOverviewUI(
        verification_id=v_id,
        risk_indicator=risk_ui,
        confidence_gauge=gauge_ui,
        claim_cards=claim_cards,
        evidence_cards=evidence_cards,
        consensus_view=consensus_views,
        timeline=timeline,
        network_graph=NetworkGraphUI(nodes=nodes_ui, edges=edges_ui),
        recommendations=explanation.actionable_recommendations,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return FullVerificationResponse(
        verification_id=v_id,
        text=request.text,
        hallucisense_score=hscore_res,
        explanation=explanation.model_dump(),
        dashboard_ui=dashboard_ui,
        execution_time_ms=round(elapsed_ms, 2),
    )
