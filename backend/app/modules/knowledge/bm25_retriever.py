"""
BM25 Sparse Retriever.
Uses rank-bm25 for lexical search.
"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    Keyword-based retrieval using BM25.
    Expects a list of dictionaries, where each dict has at least 'text' and 'title'.
    """
    def __init__(self, documents: List[Dict[str, Any]] = None):
        self.documents = documents or []
        self.tokenized_corpus = []
        self.bm25 = None
        
        if self.documents:
            self._build_index()
            
    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenizer for BM25. 
        # For production, consider using nltk or spacy.
        return text.lower().split()
        
    def _build_index(self):
        self.tokenized_corpus = [
            self._tokenize(doc.get("text", "")) for doc in self.documents
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add new documents and rebuild the index."""
        self.documents.extend(documents)
        self._build_index()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for top_k documents matching the query.
        Returns a list of dicts with 'document' and 'score'.
        """
        if not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices sorted by score descending
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            score = doc_scores[idx]
            if score > 0: # Only return matching docs
                results.append({
                    "document": self.documents[idx],
                    "score": score
                })
                
        return results
