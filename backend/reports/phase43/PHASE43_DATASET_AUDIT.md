# Phase 43.3 — Sealed 500-Case Dataset Integrity Audit

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 43.3 — Dataset Composition & Leakage Audit  
**Sample Count:** 500 Sealed Test Cases  
**Date:** 2026-09-01  

---

## 1. Modality Distribution

| Modality Category | Total Cases | Factual ($y=0$) | Hallucinated ($y=1$) | Expected Routing |
|---|---|---|---|---|
| **TEXTUAL_FACT** | 100 | 50 | 50 | Wikipedia + DeBERTa NLI |
| **ARITHMETIC** | 75 | 37 | 38 | Safe AST Symbolic Verifier |
| **UNIT_CONVERSION** | 75 | 38 | 37 | Physical Unit Verifier |
| **TEMPORAL** | 75 | 37 | 38 | Calendar & Temporal Verifier |
| **ENTITY_RELATIONSHIP** | 50 | 25 | 25 | Wikipedia + DeBERTa NLI |
| **SCIENTIFIC** | 50 | 25 | 25 | Wikipedia + DeBERTa NLI |
| **HISTORICAL** | 50 | 25 | 25 | Wikipedia + DeBERTa NLI |
| **MULTI_CLAIM** | 25 | 13 | 12 | Decomposed Multi-Pair NLI |
| **Total** | **500** | **250** | **250** | Balanced Sealed Holdout |

---

## 2. Integrity Verification

- **Exact Duplicate Overlap:** 0 records.
- **Leakage Status:** Sealed partition isolated from training and threshold tuning.
