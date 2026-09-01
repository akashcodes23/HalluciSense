# Phase 44.3 — Evidence Provenance & Audit Trail Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 44.3 — Provenance Data Structures  
**Date:** 2026-09-01  

---

## 1. Provenance Schema

Each claim verification includes:
- `source_title`: Wikipedia page title or Symbolic Parser ID.
- `source_url`: URL provenance if retrieved from network.
- `retrieved_at_utc`: Canonical UTC retrieval timestamp.
- `snippet`: Extracted passage content.
- `nli_entailment`, `nli_contradiction`, `nli_neutral`: Softmax probabilities from DeBERTa-v3 cross-encoder.
