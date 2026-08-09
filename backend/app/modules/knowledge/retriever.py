"""
Hybrid Knowledge Retriever.
Combines Wikipedia, FAISS, and BM25 strategies to fetch relevant evidence.
"""
from typing import List
import structlog
from app.modules.knowledge.wikipedia import WikipediaKnowledgeSource
from app.modules.knowledge.faiss_store import FAISSVectorStore
from app.modules.knowledge.bm25_retriever import BM25Retriever
from app.modules.knowledge.cross_encoder import CrossEncoderReranker

logger = structlog.get_logger(__name__)

class HybridRetriever:
    """Orchestrates multiple knowledge sources to find evidence for claims."""
    def __init__(self):
        self.wiki = WikipediaKnowledgeSource(max_results=3)
        self.vector_store = FAISSVectorStore()
        internal_docs = [
            {"title": "Internal Company Policy", "url": "https://intranet/policy", "text": "All employees must complete compliance training by Q3."},
            {"title": "Product Architecture", "url": "https://wiki/arch", "text": "The backend uses FastAPI and Celery for async processing."},
            {"title": "HalluciSense Design", "url": "https://wiki/design", "text": "HalluciSense uses a three-pillar system: Factual Error, Confidence Gap, and Consistency Failure."}
        ]
        self.bm25 = BM25Retriever(internal_docs)
        self.reranker = CrossEncoderReranker()
        self.last_timings = {}
        self.last_cache_metrics = {}

    def retrieve(self, claims: List[str]) -> List[dict]:
        """Retrieve evidence for all claims, batching external Wikipedia work."""
        import time
        all_evidence = []
        t_start = time.perf_counter()
        clean_claims = [c.strip() for c in claims if c and c.strip()]

        t_w0 = time.perf_counter()
        wiki_by_claim = self.wiki.retrieve_batch(clean_claims)
        wiki_ms = (time.perf_counter() - t_w0) * 1000.0
        for claim in clean_claims:
            for item in wiki_by_claim.get(claim, []):
                evidence = dict(item)
                evidence["claim"] = claim
                all_evidence.append(evidence)

        faiss_ms = 0.0
        bm25_ms = 0.0
        for claim in clean_claims:
            if self.vector_store.documents:
                t0 = time.perf_counter()
                faiss_results = self.vector_store.search(claim, top_k=2)
                faiss_ms += (time.perf_counter() - t0) * 1000.0
                for doc, sim in faiss_results:
                    all_evidence.append({"claim": claim, "source_name": doc.get("title", "Internal KB (FAISS)"), "source_url": doc.get("url", ""), "snippet": doc.get("text", ""), "similarity_score": float(sim)})

            t0 = time.perf_counter()
            bm25_results = self.bm25.search(claim, top_k=2)
            bm25_ms += (time.perf_counter() - t0) * 1000.0
            for r in bm25_results:
                doc = r["document"]
                all_evidence.append({"claim": claim, "source_name": doc.get("title", "Internal KB (BM25)"), "source_url": doc.get("url", ""), "snippet": doc.get("text", ""), "similarity_score": float(r.get("score", 0.0))})

        seen = set()
        unique_evidence = []
        for ev in all_evidence:
            key = ((ev.get("claim") or "").lower(), ev.get("snippet", ""))
            if key not in seen and ev.get("snippet", "").strip():
                seen.add(key)
                ev["is_supporting"] = True
                unique_evidence.append(ev)

        t_r0 = time.perf_counter()
        primary_claim = clean_claims[0] if clean_claims else ""
        top_evidence = self.reranker.rerank(primary_claim, unique_evidence, top_k=5) if primary_claim else []
        rerank_ms = (time.perf_counter() - t_r0) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        self.last_cache_metrics = dict(getattr(self.wiki, "last_metrics", {}))
        self.last_timings = {
            "wikipedia_ms": round(wiki_ms, 2), "faiss_ms": round(faiss_ms, 2), "bm25_ms": round(bm25_ms, 2),
            "reranker_ms": round(rerank_ms, 2), "external_retrieval_ms": round(total_ms, 2),
            "retrieval_bm25_ms": round(bm25_ms, 2), "retrieval_dense_ms": round(wiki_ms + faiss_ms, 2),
            "retrieval_hybrid_fusion_ms": round(rerank_ms, 2), "retrieval_total_ms": round(total_ms, 2),
        }
        logger.info("retrieval_completed", claims=len(clean_claims), evidence=len(top_evidence), cache=self.last_cache_metrics, timings=self.last_timings)
        return top_evidence

    def get_evidence(self, query: str) -> List[dict]:
        """Retrieve evidence for a single query with cache support."""
        if not hasattr(self, "_query_cache"):
            self._query_cache = {}
        key = query.strip().lower() if query else ""
        if key in self._query_cache:
            return self._query_cache[key]
        res = self.retrieve([query]) if query else []
        self._query_cache[key] = res
        return res
