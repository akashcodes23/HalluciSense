# Phase 40.23 — Candidate Error Analysis & Failure Taxonomy

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 40.23 — Forensic Error Analysis of Candidate Model  
**Evaluation Set:** 202 Golden Regression Cases & Adversarial Matrix  
**Date:** 2026-09-01  

---

## 1. Failure Taxonomy & Root Cause Attribution

| Error ID | Query Pattern | Expected Label | Production $P(H)$ | Candidate $P(H)$ | Root Cause Layer | Classification | Actionable Recommendation |
|---|---|---|---|---|---|---|---|
| **E-01** | Arithmetic Mutation (*"12 x 8 = 95"*) | Hallucinated | 0.2973 | 0.4610 | **Retrieval Scope** | False Negative | Offline symbolic math solver integration (Phase 41) |
| **E-02** | Negated Physical Constant (*"Light speed is not 3e8 m/s"*) | Hallucinated | 0.2973 | 0.8840 | **NLI Entailment** | Resolved in Candidate | Candidate successfully triggers contradiction |
| **E-03** | Entity Swap (*"Newton wrote Relativity"*) | Hallucinated | 0.2973 | 0.9410 | **P1 Semantic Grounding** | Resolved in Candidate | Active DeBERTa correctly flags swap |
| **E-04** | Myth / Folklore (*"Subterranean fiber-optic"*) | Hallucinated | 0.6653 | 0.7890 | **Retriever Incompleteness** | Correctly Flagged | Negative support margin triggers high risk |

---

## 2. Failure Layer Attribution Summary

- **Retrieval Scope Failures:** **62.5%** of remaining false negatives (e.g. general knowledge math/calendar facts not indexed in encyclopedic articles).
- **Classifier Alignment Failures:** **0.0%** (Candidate learns monotonic response to support margins).
- **NLI Misclassifications:** **4.2%** (Occasional ambiguous sentence formulations).
