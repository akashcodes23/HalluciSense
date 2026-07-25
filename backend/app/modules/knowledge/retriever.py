"""
Hybrid Knowledge Retriever.
Combines Wikipedia, FAISS, and BM25 strategies to fetch relevant evidence.
"""
from typing import List
from app.modules.knowledge.wikipedia import WikipediaKnowledgeSource
from app.modules.knowledge.faiss_store import FAISSVectorStore
from app.modules.knowledge.bm25_retriever import BM25Retriever
from app.modules.knowledge.cross_encoder import CrossEncoderReranker

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
        all_evidence = []
        
        for claim in claims:
            # 1. Fetch from Wikipedia (External Factual)
            wiki_results = self.wiki.retrieve(claim)
            for w in wiki_results:
                all_evidence.append(w)
                
            # 2. Fetch from Internal FAISS Vector Store (Dense Retrieval)
            if self.vector_store.documents:
                faiss_results = self.vector_store.search(claim, top_k=2)
                for doc, sim in faiss_results:
                    all_evidence.append({
                        "source_name": doc.get("title", "Internal KB (FAISS)"),
                        "source_url": doc.get("url", ""),
                        "snippet": doc.get("text", "")
                    })
                    
            # 3. Fetch from Internal BM25 Store (Sparse Retrieval)
            bm25_results = self.bm25.search(claim, top_k=2)
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
            return []
            
        # We rerank based on the first claim for simplicity in Sprint 3
        # Ideally, we'd rerank per claim and combine
        primary_claim = claims[0]
        top_evidence = self.reranker.rerank(primary_claim, unique_evidence, top_k=5)
        
        return top_evidence
