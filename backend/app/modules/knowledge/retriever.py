"""
Hybrid Knowledge Retriever.
Combines Wikipedia, FAISS, and BM25 strategies to fetch relevant evidence.
"""
from typing import List
import re
import structlog
from app.modules.knowledge.wikipedia import WikipediaKnowledgeSource
from app.modules.knowledge.faiss_store import FAISSVectorStore
from app.modules.knowledge.bm25_retriever import BM25Retriever
from app.modules.knowledge.cross_encoder import CrossEncoderReranker

logger = structlog.get_logger(__name__)

class HybridRetriever:
    """Orchestrates multiple knowledge sources to find evidence for claims."""
    def __init__(self, internal_docs: List[dict] = None):
        self.wiki = WikipediaKnowledgeSource(max_results=3)
        self.vector_store = FAISSVectorStore()
        self.bm25 = BM25Retriever(internal_docs or [])
        self.reranker = None
        self.last_timings = {}
        self.last_cache_metrics = {}

    def retrieve(self, claims: List[str]) -> List[dict]:
        """Retrieve evidence for all claims, batching external Wikipedia work."""
        import time
        from app.core.config import settings

        all_evidence = []
        t_start = time.perf_counter()
        if isinstance(claims, str):
            claims = [claims]
        clean_claims = [c.strip() for c in claims if c and c.strip()]

        t_w0 = time.perf_counter()
        wiki_by_claim = self.wiki.retrieve_batch(clean_claims)
        wiki_ms = (time.perf_counter() - t_w0) * 1000.0
        _STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for", "with", "by", "from", "that", "this", "it", "and", "or", "as"}
        for claim in clean_claims:
            claim_words = {w for w in re.findall(r'[a-zA-Z0-9]+', claim.lower()) if w not in _STOPWORDS}
            for item in wiki_by_claim.get(claim, []):
                evidence = dict(item)
                evidence["claim"] = claim
                if not evidence.get("retrieval_query"):
                    evidence["retrieval_query"] = claim
                evidence["source_name"] = item.get("source_name") or item.get("title") or "Wikipedia"
                evidence["source_url"] = item.get("source_url") or item.get("url") or ""
                if "similarity_score" not in evidence or evidence["similarity_score"] == 0.85:
                    snip_words = {w for w in re.findall(r'[a-zA-Z0-9]+', (item.get("snippet") or "").lower()) if w not in _STOPWORDS}
                    if claim_words:
                        overlap = len(claim_words.intersection(snip_words)) / float(len(claim_words))
                        evidence["similarity_score"] = round(max(0.10, min(0.95, 0.20 + 0.75 * overlap)), 4)
                    else:
                        evidence["similarity_score"] = 0.50
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
        enable_reranker = bool(getattr(settings, "HALLUCISENSE_ENABLE_RERANKER", False))
        if enable_reranker and primary_claim:
            if self.reranker is None:
                self.reranker = CrossEncoderReranker()
            top_evidence = self.reranker.rerank(primary_claim, unique_evidence, top_k=8)
        else:
            # Strongest candidates selection using existing retrieval similarity scores
            sorted_evidence = sorted(
                unique_evidence,
                key=lambda x: float(x.get("similarity_score", 0.0)),
                reverse=True,
            )
            # When multiple distinct claims are present, allocate evidence proportionally
            # so one claim does not starve the others
            if len(clean_claims) > 1:
                per_claim_quota = max(2, 8 // len(clean_claims))
                top_evidence = []
                by_claim = {}
                for ev in sorted_evidence:
                    c = ev.get("claim", "")
                    by_claim.setdefault(c, []).append(ev)
                for c, items in by_claim.items():
                    top_evidence.extend(items[:per_claim_quota])
                for ev in sorted_evidence:
                    if len(top_evidence) >= 8:
                        break
                    if ev not in top_evidence:
                        top_evidence.append(ev)
            else:
                top_evidence = sorted_evidence[:8]

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
        from collections import OrderedDict
        if not hasattr(self, "_query_cache"):
            self._query_cache = OrderedDict()
        key = query.strip().lower() if query else ""
        if key in self._query_cache:
            return list(self._query_cache[key])
        res = self.retrieve([query]) if query else []
        if len(self._query_cache) >= 512:
            self._query_cache.popitem(last=False)
        self._query_cache[key] = res
        return res
