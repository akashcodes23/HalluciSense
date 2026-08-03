"""
HalluciSense Pillar 2 — Claim Extraction Schemas
=================================================
Pydantic schemas for atomic claim decomposition, entity mentions, and relation triples.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    DECLARATIVE = "DECLARATIVE"
    NUMERICAL = "NUMERICAL"
    TEMPORAL = "TEMPORAL"
    ENTITY_RELATION = "ENTITY_RELATION"
    SCIENTIFIC = "SCIENTIFIC"


class CharacterOffsets(BaseModel):
    start: int = Field(..., description="Start character index in source text")
    end: int = Field(..., description="End character index in source text")


class ExtractedClaim(BaseModel):
    claim_id: str = Field(..., description="Unique deterministic identifier for extracted claim")
    claim_text: str = Field(..., description="Cleaned atomic claim statement")
    claim_type: ClaimType = Field(default=ClaimType.DECLARATIVE, description="Categorical type of claim")
    entities: list[str] = Field(default_factory=list, description="Extracted named entities")
    relations: list[str] = Field(default_factory=list, description="Extracted relational predicates")
    numbers: list[str] = Field(default_factory=list, description="Extracted numerical values or quantities")
    dates: list[str] = Field(default_factory=list, description="Extracted dates, years, or temporal expressions")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    sentence_index: int = Field(default=0, ge=0, description="0-indexed sentence position in original response")
    character_offsets: CharacterOffsets = Field(..., description="Span offsets in original text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extensible metadata attributes")


class ClaimExtractionRequest(BaseModel):
    text: str = Field(default="", description="LLM response text to decompose into atomic claims")
    domain_context: Optional[str] = Field(None, description="Optional domain hint (e.g., scientific, medical, general)")


class ClaimExtractionResponse(BaseModel):
    extracted_claims: list[ExtractedClaim] = Field(..., description="List of decomposed atomic claims")
    total_claims: int = Field(..., description="Total number of claims extracted")
    num_sentences: int = Field(..., description="Total sentences parsed in text")
    extraction_time_ms: float = Field(..., description="Latency in milliseconds")
