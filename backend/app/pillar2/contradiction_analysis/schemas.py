"""
HalluciSense Pillar 2 — Contradiction Analysis Schemas
======================================================
Pydantic schemas for contradiction taxonomy, contradiction items, and contradiction graph structures.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ContradictionType(str, Enum):
    DIRECT_CONTRADICTION = "DIRECT_CONTRADICTION"
    PARTIAL_CONTRADICTION = "PARTIAL_CONTRADICTION"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    FABRICATION = "FABRICATION"
    SPECULATION = "SPECULATION"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"


class ContradictionItem(BaseModel):
    contradiction_id: str = Field(..., description="Unique contradiction identifier")
    claim_id: str = Field(..., description="Claim ID involved in contradiction")
    claim_text: str = Field(..., description="Text of disputed claim")
    type: ContradictionType = Field(..., description="Categorical contradiction classification")
    severity: float = Field(..., ge=0.0, le=1.0, description="Severity score (1.0 = direct fabrication)")
    evidence_id: Optional[str] = Field(None, description="Conflicting evidence ID if present")
    explanation: str = Field(..., description="Human-readable explanation of contradiction")


class ContradictionGraphVisualization(BaseModel):
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes (Claims & Evidence)")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges (CONTRADICTS, SUPPORTS)")
    total_contradictions: int = Field(..., description="Total contradiction count")
    high_severity_count: int = Field(..., description="Count of severity >= 0.7 contradictions")


class ContradictionAnalysisResult(BaseModel):
    contradictions: List[ContradictionItem] = Field(..., description="List of detected contradictions")
    contradiction_count: int = Field(..., description="Total contradiction count")
    fabrication_index: float = Field(..., ge=0.0, le=1.0, description="Proportion of claims identified as fabrication")
    max_severity: float = Field(..., ge=0.0, le=1.0, description="Highest severity score detected")
    graph_visualization: ContradictionGraphVisualization = Field(..., description="Graph visualization JSON payload")
