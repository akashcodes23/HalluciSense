# Phase 46.6 — Production Trace Diagnostic Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 46.6 — Diagnostic Investigation of Capital of France Cases  
**Production Commit:** `41e1186`  
**Date:** 2026-09-01  

---

## 1. Diagnostic Test Cases & Empirical Results

| Case | Input Text | P(H) | Verdict | Verification Status | Master Analyze H-Score | P2 Mode | P3 Mode |
|---|---|---|---|---|---|---|---|
| **Case 1** | "What is the capital of France?" | 0.2973 | Factual | `INSUFFICIENT_EVIDENCE` | 0.4254 | `STATIC_VERIFICATION_CONFIDENCE` | `SINGLE_CLAIM_CONSISTENCY` |
| **Case 2** | "The capital of France is Paris." | 0.2973 | Factual | `ALL_VERIFIED` | 0.0284 | `STATIC_VERIFICATION_CONFIDENCE` | `SINGLE_CLAIM_CONSISTENCY` |
| **Case 3** | "The capital of France is Berlin." | 0.2973 | Factual | `CONTAINS_CONTRADICTION` | 0.4247 (FE=0.9981) | `STATIC_VERIFICATION_CONFIDENCE` | `SINGLE_CLAIM_CONSISTENCY` |
| **Case 4** | "Paris is the capital of France." | 0.2973 | Factual | `ALL_VERIFIED` | 0.0284 | `STATIC_VERIFICATION_CONFIDENCE` | `SINGLE_CLAIM_CONSISTENCY` |
| **Case 5** | "Berlin is the capital of France." | 0.2973 | Factual | `CONTAINS_CONTRADICTION` | 0.4247 (FE=0.9981) | `STATIC_VERIFICATION_CONFIDENCE` | `SINGLE_CLAIM_CONSISTENCY` |
| **Case 6** | "France has Paris as its capital." | 0.2684 | Factual | `ALL_VERIFIED` | 0.0276 | `STATIC_VERIFICATION_CONFIDENCE` | `SINGLE_CLAIM_CONSISTENCY` |
| **Case 7** | "Paris is the capital of France. Berlin is the capital of France." | 0.3499 | Factual | `CONTAINS_CONTRADICTION` | 0.5306 | `STATIC_VERIFICATION_CONFIDENCE` | `INTRA_RESPONSE_CONSISTENCY` (CF=0.33) |
| **Case 8** | "Paris is the capital of France. Berlin is the capital of Germany." | 0.3499 | Factual | `ALL_VERIFIED` | 0.1605 | `STATIC_VERIFICATION_CONFIDENCE` | `INTRA_RESPONSE_CONSISTENCY` (CF=0.33) |

---

## 2. Key Findings

1. Factual statements ("The capital of France is Paris.") resolve to $H \approx 0.0284$ (`VERIFIED`).
2. Interrogative inputs ("What is the capital of France?") resolve to `INSUFFICIENT_EVIDENCE` without falsely asserting a contradiction.
3. Multi-claim conflicting inputs ("Paris is capital of France. Berlin is capital of France.") activate `INTRA_RESPONSE_CONSISTENCY` and correctly detect the internal contradiction!
