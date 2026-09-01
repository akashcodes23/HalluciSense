# Phase 40.24 — Memory Safety & Shadow Model Headroom Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 40.24 — Candidate Shadow Memory Profiling  
**Container Limit:** 1024 MB  
**Date:** 2026-09-01  

---

## 1. Memory Profile Across Execution Modes

| Operational Mode | Process RSS | Baseline RSS | Delta | Free Headroom (vs 1024 MB) |
|---|---|---|---|---|
| **Production Frozen Only** | **528.8 MB** | 528.8 MB | +0.0 MB | 495.2 MB (48.4%) |
| **Shadow Semantic NLI Enabled** | **538.0 MB** | 538.0 MB | +0.0 MB | 486.0 MB (47.5%) |
| **Dual Classifier Shadow Mode (Production + Candidate C)** | **539.8 MB** | 538.0 MB | +1.8 MB | 484.2 MB (47.3%) |
| **10 Sequential Requests** | **540.1 MB** | 538.2 MB | +1.9 MB | 483.9 MB (47.2%) |
| **2 Concurrent Requests** | **541.6 MB** | 539.6 MB | +2.0 MB | 482.4 MB (47.1%) |

---

## 2. In-Memory Artifact Footprint

- `hybrid_meta_classifier_phase40_candidate.joblib`: **218 KB**
- `preprocessing_phase40_candidate.joblib`: **799 bytes**
- `nli_model` (Singleton DeBERTa-v3): **~280 MB**
- **OOM Events:** **0**
- **Container Limit Violations:** **0**
