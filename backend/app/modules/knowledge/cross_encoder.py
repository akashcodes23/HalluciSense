"""
CrossEncoder Reranker using Singleton ModelRegistry.
Uses sentence-transformers to accurately score (query, document) pairs.
"""
from typing import List, Dict, Any
import structlog
from app.core.engine.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Reranks candidate evidence based on semantic relevance to the claim."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = ModelRegistry.get_cross_encoder_reranker(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Given a query and candidate evidence dicts, returns the top_k reranked dicts."""
        if not candidates:
            return []

        pairs = [[query, cand.get("snippet", "")] for cand in candidates]
        scores = self.model.predict(pairs)

        scored_candidates = []
        for cand, score in zip(candidates, scores):
            import math
            prob = 1 / (1 + math.exp(-score))
            cand_copy = dict(cand)
            cand_copy["similarity_score"] = float(prob)
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_candidates[:top_k]
