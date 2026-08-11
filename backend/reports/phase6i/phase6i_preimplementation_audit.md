# Phase 6I: Pre-Implementation Forensic Audit & Experimental Design Report

**Date**: 2026-08-11  
**Git SHA**: `583231e`  
**Target Mechanism**: Claim-Level Retrieval Reconstruction & Claim-Local Evidence-Date Alignment  

---

## 1. Current Architecture & Evidence Flow Audit

### Evidence Flow & Pipeline Layers
- **Pillar 1 Retrieval & Verification** (`app/core/engine/pillar1_retrieval.py`):
  - Extracts sentences from LLM response.
  - Scores NLI entailment between response sentences and retrieved evidence passages using `nli-deberta-v3-small`.
  - Aggregates evidence globally across passages ($Y_E = \bigcup y_e$).
- **Pillar 3 Temporal & Epistemic Verification** (`app/core/engine/temporal.py` & `epistemic.py`):
  - Resolves query and response epistemic frames (`EpistemicResolver`).
  - Evaluates temporal inconsistency scores ($S_{\text{temporal}}$).
  - Applies Epistemic Gate function $G(M_q, M(c_i))$ to protect non-assertions from false temporal penalties.
- **Fusion Layer** (`app/core/engine/fusion.py`):
  - Combines Pillar scores: $H = \alpha S_1 + \beta S_2 + \gamma S_3$ with $\alpha=0.40, \beta=0.30, \gamma=0.30$ (**FROZEN**).

---

## 2. Identified Evidence-Alignment Limitation from Phase 6E
- In multi-claim responses where Passage 1 contains dates for Claim 1 and Passage 2 contains dates for Claim 2, global anchor union ($Y_E = \bigcup y_e$) evaluates candidate anchors against the whole passage set.
- While $Y_E$ effectively eliminates false mismatches from background passage dates in single-claim scenarios, in multi-claim responses it can permit cross-claim date contamination (Claim 1 matching a date from Passage 2).
- **Proposed Intervention**: Claim-level retrieval reconstruction ($Y_i$), where each atomic claim $c_i$ is mapped to a claim-local evidence subset $E(c_i)$ and claim-local temporal anchors $Y_i = \text{anchors}(E(c_i))$.

---

## 3. Immutable Production Invariants
- Fusion weights: $\alpha=0.40, \beta=0.30, \gamma=0.30$ (**FROZEN**).
- Production risk thresholds: `VERIFIED < 0.35`, `NEEDS_VERIFICATION < 0.50`, `MODERATE_RISK < 0.65`, `LIKELY_HALLUCINATED >= 0.65` (**FROZEN**).
- Phase 6F locked architecture and `LOCKED_FINAL_TEST` dataset (**IMMUTABLE**).
- Zero benchmark-specific dates, entities, or heuristics permitted.

---

## 4. Phase 6I Candidate Systems (R0 -> R6)
- **R0**: Existing frozen Phase 6E baseline (Global evidence union $Y_E$).
- **R1**: Claim segmentation only (atomic claims evaluated independently against $Y_E$).
- **R2**: Claim-specific evidence selection ($E(c_i)$ filtering).
- **R3**: Claim-specific temporal anchor extraction ($Y_i = \text{anchors}(E(c_i))$).
- **R4**: Claim-specific evidence-date alignment ($S_{\text{temporal}}(c_i, Y_i)$).
- **R5**: Claim-level reconstruction + Temporal-Epistemic Gate $G(M_q, M(c_i))$.
- **R6**: Full candidate Phase 6I architecture.

---

## 5. Risk of Leakage & Computational Cost Strategy
- **Dataset Hash Check**: Phase 6I independent benchmark ($N=500$) will be verified with SHA-256 to ensure 0 overlap with Phase 6D, 6E, 6F, and `LOCKED_FINAL_TEST`.
- **Expected Overhead**: Claim-level evidence selection adds negligible CPU overhead (~0.1–0.3 ms per claim).
