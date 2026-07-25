"""
AI Provider Schemas — shared data structures for the provider abstraction layer.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

class TokenLogit(BaseModel):
    """Represents a single token and its associated log probability."""
    token: str
    logprob: float

class ProviderResponseChunk(BaseModel):
    """
    A single chunk yielded from a streaming text generation.
    """
    text: str
    logits: Optional[List[TokenLogit]] = Field(default=None)
    is_done: bool = Field(default=False)
