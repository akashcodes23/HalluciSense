"""
HalluciSense Pillar 2 — Multi-LLM Verifier Schemas
===================================================
Pydantic schemas for verification labels, provider responses, and multi-LLM verification requests.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VerificationLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNKNOWN = "UNKNOWN"


class SingleClaimVerification(BaseModel):
    claim_id: str = Field(..., description="Target claim ID")
    provider_name: str = Field(..., description="LLM provider name (e.g., Gemini, GPT-4, Claude)")
    label: VerificationLabel = Field(..., description="Normalized verification label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Provider verification confidence")
    reasoning: str = Field(..., description="Explanation string from LLM provider")
    supporting_evidence_ids: List[str] = Field(default_factory=list, description="IDs of supporting evidence items")
    contradicting_evidence_ids: List[str] = Field(default_factory=list, description="IDs of contradicting evidence items")
    latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds")


class MultiLLMVerificationRequest(BaseModel):
    claim_id: str = Field(..., description="Target claim ID")
    claim_text: str = Field(..., description="Atomic claim statement text")
    evidence_snippets: List[Dict[str, Any]] = Field(..., description="List of evidence dicts with ID and text")
    verifiers: Optional[List[str]] = Field(None, description="Target verifiers (e.g. ['Gemini', 'GPT-4']); None uses all active")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature for verifiers")


class MultiLLMVerificationResponse(BaseModel):
    claim_id: str = Field(..., description="Target claim ID")
    verifications: List[SingleClaimVerification] = Field(..., description="Per-provider verification results")
    providers_attempted: List[str] = Field(..., description="List of provider verifiers attempted")
    failed_verifiers: List[str] = Field(default_factory=list, description="List of verifiers that failed or timed out")
    total_latency_ms: float = Field(..., description="Total wall-clock latency in milliseconds")
