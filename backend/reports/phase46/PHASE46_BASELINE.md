# Phase 46.1 — Forensic Baseline: Multi-Pillar Verification & Consistency Engine

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 46.1 — Forensic Baseline Audit & Problem Framing  
**Production Commit:** `41e1186`  
**Date:** 2026-09-01  

---

## 1. System Inventory Before Phase 46

- **Production Classifier:** `HistGradientBoostingClassifier` (19 canonical features, $\tau^* = 0.54$, $N=58,002$, Frozen).
- **Candidate C:** Retained in shadow diagnostic mode only.
- **Pillar 1:** Evidence Grounding active via hybrid retrieval and DeBERTa cross-encoder NLI.
- **Evidence Intelligence Gateway:** Active for deterministic arithmetic, physical units, and temporal logic.

---

## 2. Identified Production Trace Anomalies

1. **Pillar 2 ("Token logprobs not available"):** For static `/verify` and `/predict` requests, Pillar 2 returned `available = False` because token log probabilities were absent.
   - *Resolution:* Introduce `STATIC_VERIFICATION_CONFIDENCE` mode deriving confidence from evidence coverage, NLI certainty, and retrieval relevance without fabricating logprobs.
2. **Pillar 3 ("Multi-generation not available for static input"):** For static `/verify` requests, Pillar 3 returned `available = False` because alternate LLM generations were omitted.
   - *Resolution:* Introduce `INTRA_RESPONSE_CONSISTENCY` mode executing pairwise semantic and NLI contradiction checks across multi-claim inputs, and `SINGLE_CLAIM_CONSISTENCY` for single claims.
3. **Root Cause Mislabelling ("Entity Linking Failure"):** `RootCauseClassifier` mapped all high factual errors ($FE \ge 0.80$) to `Entity Linking Failure`.
   - *Resolution:* Distinguish `FACTUAL_CONTRADICTION` from `EVIDENCE_MISSING` and `RETRIEVAL_FAILURE`.
