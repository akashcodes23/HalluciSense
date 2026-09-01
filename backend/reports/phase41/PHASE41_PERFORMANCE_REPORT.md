# Phase 41.24 — Latency & Throughput Benchmark Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 41.24 — End-to-End Latency Profile Under Shadow Execution  
**Device:** CPU (Single-threaded OpenMP/MKL capped)  
**Date:** 2026-09-01  

---

## 1. Latency Profile Breakdown

| Pipeline Stage | P50 (ms) | P95 (ms) | Notes |
|---|---|---|---|
| **Claim Segmentation** | 0.8 ms | 2.1 ms | Regex segmentation |
| **Evidence Retrieval** | 890.0 ms | 1150.0 ms | Wikipedia network retrieval |
| **Semantic NLI Grounding** | 68.0 ms | 148.0 ms | Batched DeBERTa-v3 inference |
| **Pillar 2 Pairwise Structure** | 18.0 ms | 65.0 ms | Pairwise cache + embeddings |
| **Production Inference & Attribution** | 2.5 ms | 4.8 ms | 21 classifier evaluations |
| **Candidate C Shadow Inference** | 0.3 ms | 0.8 ms | Single forward pass on scaled features |
| **Total Shadow Request Latency** | **981.0 ms** | **1372.0 ms** | Incremental shadow overhead: < 1.0 ms |

---

## 2. Throughput & Scalability

Shadow inference adds less than **0.1% latency overhead** compared to standard production inference because both models share the exact same 19-dimensional feature representation.
