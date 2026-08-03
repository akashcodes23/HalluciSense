"""
HalluciSense Pillar 2 — Evidence Retrieval Schemas
===================================================
Pydantic schemas for evidence items, citation metadata, provider configs, and retrieval requests.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CitationMetadata(BaseModel):
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    authors: List[str] = Field(default_factory=list, description="Author list")
    journal: Optional[str] = Field(None, description="Journal or publisher name")
    volume: Optional[str] = Field(None, description="Volume or issue")
    citation_count: Optional[int] = Field(0, description="Total citations")
    license: Optional[str] = Field(None, description="Publication license")


class EvidenceItem(BaseModel):
    evidence_id: str = Field(..., description="Unique evidence item identifier")
    title: str = Field(..., description="Title of reference document or entry")
    source: str = Field(..., description="Provider source identifier (e.g. Wikipedia, PubMed)")
    url: str = Field(..., description="Canonical source URL")
    snippet: str = Field(..., description="Retrieved excerpt or passage text")
    publication_date: Optional[str] = Field(None, description="Publication or revision date")
    authority_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Domain authority weighting (0-1)")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Retrieval relevance confidence")
    retrieval_latency_ms: float = Field(default=0.0, ge=0.0, description="Retrieval latency in milliseconds")
    citation_metadata: CitationMetadata = Field(default_factory=CitationMetadata, description="Bibliographic metadata")


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query or claim text to retrieve evidence for")
    max_results_per_provider: int = Field(default=3, ge=1, le=20, description="Maximum items per provider")
    providers: Optional[List[str]] = Field(None, description="Specific providers to query; None queries all active")
    timeout_seconds: float = Field(default=5.0, ge=0.5, le=30.0, description="Request timeout per provider")


class RetrievalResponse(BaseModel):
    query: str = Field(..., description="Original query")
    items: List[EvidenceItem] = Field(..., description="Retrieved evidence items")
    total_retrieved: int = Field(..., description="Total evidence item count")
    providers_queried: List[str] = Field(..., description="List of providers attempted")
    failed_providers: List[str] = Field(default_factory=list, description="List of providers that timed out or failed")
    total_latency_ms: float = Field(..., description="Total wall-clock latency in milliseconds")
