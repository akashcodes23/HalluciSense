# PHASE 53 — REPOSITORY, INVARIANTS & BASELINE AUDIT
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `FROZEN PRODUCTION INVARIANTS VERIFIED`

---

## 1. Production Artifacts & Cryptographic Checksums

| Artifact | Canonical Path | SHA256 Checksum | Invariant Status |
| :--- | :--- | :--- | :--- |
| **Hybrid Meta Classifier** | `backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib` | `089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad` | ✅ FROZEN & IMMUTABLE |
| **Robust Preprocessing Scaler**| `backend/evaluation_results/phase6m/final_hybrid_model/preprocessing.joblib` | `bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90` | ✅ FROZEN & IMMUTABLE |
| **Operating Decision Threshold**| Constant | `tau* = 0.54` | ✅ FROZEN & IMMUTABLE |
| **Canonical Feature Schema** | 19 Features (`SET_A_FULL_HYBRID`) | `feature_schema.json` | ✅ FROZEN & IMMUTABLE |

---

## 2. Phase 52 Baseline Findings Summary

- **Phase 52 Production Commit**: `2fada89`
- **P1-Only Grounding (Balanced $N=300$)**: AUROC: **0.8083**, Recall: **80.00%**, Specificity: **60.67%**, MCC: **0.4145**
- **Dual Dynamic Fusion (P1 + P2)**: AUROC: **0.8139**, Recall: **77.33%**, Specificity: **66.00%**, MCC: **0.4361**
- **Frozen 19-Feature Classifier**: AUROC: **0.6905**, Recall: **30.67%**, Specificity: **83.33%**, MCC: **0.1647**
- **Actionable Forensic Hypotheses**:
  1. Downstream 19-feature meta-classifier contains tree splits with inverted polarities on key grounding signals (`p1_mean_contradiction`, `prob_ratio`, `p2_max_pairwise_similarity`).
  2. Symbolic verification gateway is fully accurate (100%), but arithmetic/unit/temporal contradiction signals are shadow-only and suppressed from the active 19-feature vector.
