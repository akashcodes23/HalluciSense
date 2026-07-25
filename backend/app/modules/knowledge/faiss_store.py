"""
FAISS Vector Store (Mocked for Windows compatibility).
Handles dense vector indexing and similarity search using simple keyword matching for now.
"""
from typing import List, Tuple

class FAISSVectorStore:
    """
    Mock vector store for caching and searching documents.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embed_dim: int = 384):
        self.documents: List[dict] = []

    def add_documents(self, documents: List[dict]):
        if documents:
            self.documents.extend(documents)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        if not self.documents:
            return []
            
        results = []
        q_lower = query.lower()
        for doc in self.documents:
            text = doc.get("text", "").lower()
            # extremely basic mock similarity
            sim = 0.5
            if q_lower in text:
                sim = 0.9
            results.append((doc, sim))
            
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
