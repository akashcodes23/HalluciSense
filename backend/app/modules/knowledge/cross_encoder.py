"""
CrossEncoder Reranker.
Uses sentence-transformers to accurately score (query, document) pairs.
"""
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger(__name__)

class CrossEncoderReranker:
    """
    Reranks candidate evidence based on semantic relevance to the claim.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Lazy import so it doesn't block fast startup if unused
        from sentence_transformers import CrossEncoder
        logger.info("Initializing CrossEncoder model", model_name=model_name)
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Given a query and candidate evidence dicts (which must have a 'snippet' key),
        returns the top_k reranked dicts.
        """
        if not candidates:
            return []
            
        # Prepare pairs for the CrossEncoder
        pairs = [[query, cand.get("snippet", "")] for cand in candidates]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Attach scores to candidates
        scored_candidates = []
        for cand, score in zip(candidates, scores):
            # Normalizing/converting cross encoder logit to probability is optional,
            # but we can just use the raw score for sorting. 
            # Or use sigmoid if we want 0-1 range.
            import math
            prob = 1 / (1 + math.exp(-score))
            
            cand_copy = dict(cand)
            cand_copy["similarity_score"] = float(prob) # Overwrite initial retrieval score
            scored_candidates.append(cand_copy)
            
        # Sort descending
        scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_candidates[:top_k]
