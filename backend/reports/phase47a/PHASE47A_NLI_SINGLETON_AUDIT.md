# Phase 47A — NLI & Embedding Model Singleton Audit

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 47A — Singleton Model Registry Audit  
**Date:** 2026-09-01  

---

## 1. Audit Findings

1. **DeBERTa NLI:** Exactly 1 singleton instance loaded via `ModelRegistry.get_nli_model("cross-encoder/nli-deberta-v3-small")`.
2. **SentenceTransformer:** Exactly 1 singleton instance loaded via `ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2")`.
3. **CrossEncoder Reranker:** Exactly 1 singleton instance loaded via `ModelRegistry.get_cross_encoder_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")`.
4. **Pipeline Orchestrator:** Exactly 1 singleton instance loaded via `ModelRegistry.get_pipeline()`.

---

## 2. Model Counts Verified

- `init_counts["nli_model"] = 1`
- `init_counts["sentence_transformer"] = 1`
- `init_counts["cross_encoder_reranker"] = 1`
- `init_counts["pipeline"] = 1`
