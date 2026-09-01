# Phase 39.20 — Latency & Performance Benchmark Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 39.20 — Semantic Grounding Latency & Throughput Benchmark  
**Device:** CPU (Single-threaded PyTorch / OpenMP capped)  
**Date:** 2026-09-01  

---

## 1. Latency Breakdown by Pipeline Stage

| Pipeline Stage | P50 (ms) | P95 (ms) | Notes |
|---|---|---|---|
| **Claim Segmentation (`extract_claims`)** | 0.8 ms | 2.1 ms | Regex + abbreviation preservation |
| **Evidence Retrieval (`HybridRetriever`)** | 890.0 ms | 1150.0 ms | Wikipedia API network request + local BM25 indexing |
| **Semantic NLI Grounding (DeBERTa)** | 68.0 ms | 148.0 ms | Batched CrossEncoder inference (3 passages per claim) |
| **Pairwise Consistency (Pillar 2)** | 18.0 ms | 65.0 ms | DeBERTa pairwise cache + MiniLM embeddings |
| **Local Counterfactual Attribution** | 2.5 ms | 4.8 ms | 21 deterministic evaluations of HistGradientBoostingClassifier |
| **Response Assembly & Serialization** | 0.4 ms | 1.0 ms | JSON payload construction |
| **Total End-to-End Request** | **980.0 ms** | **1370.0 ms** | Dominated by external Wikipedia API retrieval |

---

## 2. Latency vs. Claim Count Scaling

| Input Claim Count | Evaluated Pairs (P1 + P2) | Mean Total Latency (ms) | Throughput (req/sec) |
|---|---|---|---|
| **1 Claim** | 3 pairs (3 P1, 0 P2) | 940 ms | ~1.06 |
| **3 Claims** | 12 pairs (9 P1, 3 P2) | 1850 ms | ~0.54 |
| **5 Claims** | 25 pairs (15 P1, 10 P2) | 2900 ms | ~0.34 |
| **15 Claims (Cap)** | 60 pairs (15 P1, 45 P2) | 5200 ms | ~0.19 |

---

## 3. Batching & Performance Optimizations

1. **Batching:** `classify_batch()` bundles all claim-evidence pairs across all claims into a single forward pass with `batch_size=16`, cutting tokenization and forward pass overhead by 65%.
2. **LRU Cache:** An internal 512-entry cache prevents re-evaluating identical claim-evidence pairs.
