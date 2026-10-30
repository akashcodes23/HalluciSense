"""
AI Provider Schemas — shared data structures for the provider abstraction layer.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class TokenMeasurement(BaseModel):
    """Represents a single token and its generation-time probability measurement with full provenance."""
    token: str = Field(..., description="Selected token text")
    logprob: float = Field(..., description="Natural log-probability ln(p) <= 0.0 of selected token")
    token_id: Optional[int] = Field(default=None, description="Vocabulary token index if provided by model")
    position: Optional[int] = Field(default=None, description="Zero-based token sequence position")
    provider: Optional[str] = Field(default=None, description="LLM provider name (e.g. Google Gemini, OpenAI)")
    model: Optional[str] = Field(default=None, description="Model identifier (e.g. gemini-1.5-flash, gpt-4o)")
    request_id: Optional[str] = Field(default=None, description="Provider request ID if provided")
    top_logprobs: Optional[List[Dict[str, Any]]] = Field(default=None, description="Alternative candidate logprobs")

class TokenLogit(TokenMeasurement):
    """Backward-compatible alias for TokenMeasurement."""
    pass

class ProviderResponseChunk(BaseModel):
    """
    A single chunk yielded from a streaming text generation.
    """
    text: str
    logits: Optional[List[TokenLogit]] = Field(default=None)
    is_done: bool = Field(default=False)
