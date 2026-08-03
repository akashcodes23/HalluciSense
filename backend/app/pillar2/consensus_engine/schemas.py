"""
HalluciSense Pillar 2 — Consensus Engine Schemas
=================================================
Pydantic schemas for multi-verifier label consensus, agreement matrices, and disagreement metrics.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.pillar2.multi_llm_verifier.schemas import SingleClaimVerification, VerificationLabel


class DisagreementDetail(BaseModel):
    verifier_name: str = Field(..., description="Dissenting verifier name")
    assigned_label: VerificationLabel = Field(..., description="Label assigned by dissenting verifier")
    confidence: float = Field(..., description="Confidence of dissenting verifier")
    reasoning: str = Field(..., description="Rationale for dissent")


class ConsensusResult(BaseModel):
    claim_id: str = Field(..., description="Target claim ID")
    majority_label: VerificationLabel = Field(..., description="Majority vote consensus label")
    weighted_label: VerificationLabel = Field(..., description="Confidence-weighted consensus label")
    consensus_confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregated consensus confidence")
    label_distribution: Dict[str, int] = Field(..., description="Raw label counts across verifiers")
    label_weights: Dict[str, float] = Field(..., description="Summed confidence weights per label")
    pairwise_agreement_score: float = Field(..., ge=0.0, le=1.0, description="Mean pairwise agreement ratio (0-1)")
    shannon_entropy: float = Field(..., ge=0.0, description="Shannon entropy of label distribution (lower = more consensus)")
    confidence_variance: float = Field(..., ge=0.0, description="Variance in confidence scores across verifiers")
    agreement_matrix: Dict[str, Dict[str, float]] = Field(..., description="Pairwise agreement matrix between verifiers")
    disagreeing_verifiers: List[DisagreementDetail] = Field(default_factory=list, description="List of dissenting verifiers")
    verdict_summary: str = Field(..., description="Human-readable consensus summary")
