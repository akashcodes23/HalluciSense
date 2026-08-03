"""
HalluciSense Pillar 2 — Unified H-Score Schemas
================================================
Pydantic schemas for the next-generation HalluciSense Score (0-100), risk levels, and breakdown components.
"""

from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field


class RiskCategory(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class UnifiedHScoreResult(BaseModel):
    hallucisense_score: float = Field(..., ge=0.0, le=100.0, description="Unified HalluciSense Score (0-100)")
    risk_category: RiskCategory = Field(..., description="Categorical risk classification")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall verification confidence (0-1)")
    pillar1_probability: float = Field(..., ge=0.0, le=1.0, description="Frozen Pillar 1 statistical probability")
    evidence_score: float = Field(..., ge=0.0, le=100.0, description="Pillar 2 Evidence-level risk score (0-100)")
    consensus_score: float = Field(..., ge=0.0, le=100.0, description="Pillar 2 Consensus-level risk score (0-100)")
    contradiction_score: float = Field(..., ge=0.0, le=100.0, description="Pillar 2 Contradiction-level risk score (0-100)")
    component_weights: Dict[str, float] = Field(..., description="Weight allocation across score components")
    explanation_summary: str = Field(..., description="Concise human-readable rationale for score")
