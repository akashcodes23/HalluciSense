# Phase 45.1 — Forensic Baseline: Final Red-Team & Viva Acceptance

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 45.1 — Forensic Baseline Audit & State Freeze  
**Production Commit:** `17f31ed`  
**Date:** 2026-09-01  

---

## 1. Frozen Production Baseline Snapshot

| Parameter | Frozen Production Contract | SHA256 / Checksum Status |
|---|---|---|
| **Production Classifier** | `HistGradientBoostingClassifier` | `089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad` |
| **RobustScaler** | 19 Features (`preprocessing.joblib`) | `bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90` |
| **Decision Threshold** | $\tau^* = 0.54$ | Hardcoded in `pipeline.py` & verified |
| **Feature Schema** | 19 Canonical Features (SET_A_FULL_HYBRID) | Immutable |
| **Evidence Gateway** | `EvidenceIntelligenceGateway` (Symbolic, Unit, Temporal, NLI) | Promoted in Phase 43 |
| **Shadow Candidate** | `phase40_candidate_v1` | Retained in shadow mode only |
| **Railway Container Memory** | 1024 MB Limit | Measured at ~538 MB steady RSS |

---

## 2. Test Suite & Frontend Health

- **Backend Tests:** 147 tests across Phase 37 through Phase 44 (100% PASS).
- **Frontend Build:** 0 TypeScript errors, 23 static prerendered Next.js routes.
- **Explainability:** Exact counterfactual attribution ($a_i = P(H|X) - P(H|X_i)$) + Human-auditable `VerificationTracePanel`.
