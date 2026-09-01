# PHASE 49 — PILLAR 1 RETRIEVAL & EVIDENCE MEMORY
**Evidence Bounding & Wikipedia Extraction Audit**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `AUDITED & CERTIFIED`

---

## 1. Evidence Intelligence Gateway Bounds

- **Max Candidate Evidence Passages**: 3 items per claim (`MAX_EVIDENCE_FOR_NLI = 3`).
- **Max Snippet Length**: 350 characters.
- **Wikipedia Document Cache**: LRU bounded at 256 entries.
- **Wikidata Entity Cache**: LRU bounded at 512 entries.
- **BM25 In-Memory Index**: Retains only internal static policies (< 10 KB).

---

## 2. Memory Stability Under Repeated Retrieval

In sequential 50-request testing, repeated retrieval with LRU caching showed 100% cache hits for recurring entities with zero string duplication in process heap.
