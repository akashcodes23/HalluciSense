"""
Hybrid Knowledge Retriever.
Combines Wikipedia, FAISS, and BM25 strategies to fetch relevant evidence.
"""
from typing import List
from app.modules.knowledge.wikipedia import WikipediaKnowledgeSource
from app.modules.knowledge.faiss_store import FAISSVectorStore
from app.modules.knowledge.bm25_retriever import BM25Retriever
from app.modules.knowledge.cross_encoder import CrossEncoderReranker
import structlog

logger = structlog.get_logger(__name__)

class HybridRetriever:
    """
    Orchestrates multiple knowledge sources to find evidence for a claim.
    """
    def __init__(self):
        self.wiki = WikipediaKnowledgeSource(max_results=3)
        self.vector_store = FAISSVectorStore()
        
        # Load some mock documents for BM25 and FAISS internal search for Sprint 3
        internal_docs = [
            {"title": "Internal Company Policy", "url": "https://intranet/policy", "text": "All employees must complete compliance training by Q3."},
            {"title": "Product Architecture", "url": "https://wiki/arch", "text": "The backend uses FastAPI and Celery for async processing."},
            {"title": "HalluciSense Design", "url": "https://wiki/design", "text": "HalluciSense uses a three-pillar system: Factual Error, Confidence Gap, and Consistency Failure."}
        ]
        
        self.bm25 = BM25Retriever(internal_docs)
        self.reranker = CrossEncoderReranker()

    def retrieve(self, claims: List[str]) -> List[dict]:
        """
        Given a list of claims (or a single text broken into claims),
        retrieve relevant evidence snippets from all configured sources.
        """
        import time
        all_evidence = []
        
        t_ext_start = time.perf_counter()
        wiki_total_ms = 0.0
        faiss_total_ms = 0.0
        bm25_total_ms = 0.0
        rerank_total_ms = 0.0

        for claim in claims:
            logger.info("retrieving_evidence_for_claim", claim=claim)
            # 1. Fetch from Wikipedia (External Factual)
            t_w0 = time.perf_counter()
            wiki_results = self.wiki.retrieve(claim)
            wiki_total_ms += (time.perf_counter() - t_w0) * 1000.0
            for w in wiki_results:
                all_evidence.append(w)
                
            # 2. Fetch from Internal FAISS Vector Store (Dense Retrieval)
            if self.vector_store.documents:
                t_f0 = time.perf_counter()
                faiss_results = self.vector_store.search(claim, top_k=2)
                faiss_total_ms += (time.perf_counter() - t_f0) * 1000.0
                for doc, sim in faiss_results:
                    all_evidence.append({
                        "source_name": doc.get("title", "Internal KB (FAISS)"),
                        "source_url": doc.get("url", ""),
                        "snippet": doc.get("text", "")
                    })
                    
            # 3. Fetch from Internal BM25 Store (Sparse Retrieval)
            t_b0 = time.perf_counter()
            bm25_results = self.bm25.search(claim, top_k=2)
            bm25_total_ms += (time.perf_counter() - t_b0) * 1000.0
            for r in bm25_results:
                doc = r["document"]
                all_evidence.append({
                    "source_name": doc.get("title", "Internal KB (BM25)"),
                    "source_url": doc.get("url", ""),
                    "snippet": doc.get("text", "")
                })
                    
        # Simple deduplication by snippet text before reranking
        seen = set()
        unique_evidence = []
        for ev in all_evidence:
            snippet = ev["snippet"]
            if snippet not in seen:
                seen.add(snippet)
                ev["is_supporting"] = True
                unique_evidence.append(ev)
                
        # 4. Rerank all candidates using CrossEncoder
        if not claims:
            self.last_timings = {
                "wikipedia_ms": round(wiki_total_ms, 2),
                "faiss_ms": round(faiss_total_ms, 2),
                "bm25_ms": round(bm25_total_ms, 2),
                "reranker_ms": 0.0,
                "external_retrieval_ms": round((time.perf_counter() - t_ext_start) * 1000.0, 2),
                "retrieval_bm25_ms": round(bm25_total_ms, 2),
                "retrieval_dense_ms": round(wiki_total_ms + faiss_total_ms, 2),
                "retrieval_hybrid_fusion_ms": 0.0,
            }
            return []

        primary_claim = claims[0]
        t_r0 = time.perf_counter()
        top_evidence = self.reranker.rerank(primary_claim, unique_evidence, top_k=5)
        rerank_total_ms = (time.perf_counter() - t_r0) * 1000.0

        ext_total_ms = (time.perf_counter() - t_ext_start) * 1000.0
        dense_total_ms = wiki_total_ms + faiss_total_ms

        self.last_timings = {
            "wikipedia_ms": round(wiki_total_ms, 2),
            "faiss_ms": round(faiss_total_ms, 2),
            "bm25_ms": round(bm25_total_ms, 2),
            "reranker_ms": round(rerank_total_ms, 2),
            "external_retrieval_ms": round(ext_total_ms, 2),
            "retrieval_bm25_ms": round(bm25_total_ms, 2),
            "retrieval_dense_ms": round(dense_total_ms, 2),
            "retrieval_hybrid_fusion_ms": round(rerank_total_ms, 2),
        }
        return top_evidence

    def get_evidence(self, query: str) -> List[dict]:
        """Retrieve evidence passages for a single query text."""
        if not hasattr(self, "_query_cache"):
            self._query_cache = {}
        if query in self._query_cache:
            return self._query_cache[query]
        res = self.retrieve([query]) if query else []
        self._query_cache[query] = res
        return res

