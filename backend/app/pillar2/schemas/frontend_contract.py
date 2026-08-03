"""
HalluciSense Pillar 2 — Frontend Contract Schemas
=================================================
Backend-ready Pydantic schemas for frontend UI components (Dashboard, Evidence Cards,
Claim Cards, Consensus View, Risk Indicators, Timeline, Confidence Gauges, Network Graph).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.pillar2.unified_hscore.schemas import RiskCategory


class ClaimCardUI(BaseModel):
    claim_id: str
    text: str
    claim_type: str
    consensus_label: str
    confidence_percentage: float
    supporting_sources_count: int
    contradicting_sources_count: int


class EvidenceCardUI(BaseModel):
    evidence_id: str
    title: str
    provider: str
    url: str
    snippet: str
    authority_score: float
    publication_date: Optional[str]


class ConsensusViewUI(BaseModel):
    claim_id: str
    majority_label: str
    agreement_percentage: float
    shannon_entropy: float
    label_breakdown: Dict[str, int]
    dissenting_verifiers: List[str]


class RiskIndicatorUI(BaseModel):
    hallucisense_score: float
    risk_category: RiskCategory
    color_hex: str
    badge_label: str


class ConfidenceGaugeUI(BaseModel):
    overall_confidence: float
    confidence_level: str
    pillar1_probability: float
    evidence_quality_score: float


class TimelineItemUI(BaseModel):
    step_name: str
    status: str
    latency_ms: float
    description: str


class GraphNodeUI(BaseModel):
    id: str
    label: str
    type: str
    category: Optional[str] = None


class GraphEdgeUI(BaseModel):
    source: str
    target: str
    relation: str
    severity: Optional[float] = None


class NetworkGraphUI(BaseModel):
    nodes: List[GraphNodeUI]
    edges: List[GraphEdgeUI]


class DashboardOverviewUI(BaseModel):
    verification_id: str
    risk_indicator: RiskIndicatorUI
    confidence_gauge: ConfidenceGaugeUI
    claim_cards: List[ClaimCardUI]
    evidence_cards: List[EvidenceCardUI]
    consensus_view: List[ConsensusViewUI]
    timeline: List[TimelineItemUI]
    network_graph: NetworkGraphUI
    recommendations: List[str]
