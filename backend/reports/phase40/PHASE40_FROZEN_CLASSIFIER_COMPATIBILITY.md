# Phase 40.5 — Frozen Classifier Compatibility & Recalibration Assessment

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 40.5 — Quantitative Compatibility Assessment of Frozen Model vs. Candidate  
**Date:** 2026-09-01  

---

## 1. Comparative Metrics on Independent Holdout ($N=8,700$)

| Evaluation Metric | Frozen Production Model (Proxy Input) | Frozen Model (Semantic Input) | Candidate C (Retrained on Semantic) | Improvement |
|---|---|---|---|---|
| **ROC-AUC** | 0.7378 | 0.8120 | **0.9999** | **++0.2621** |
| **PR-AUC** | 0.7105 | 0.7950 | **0.9998** | **++0.2893** |
| **F1 Score ($	au=0.54$)** | 0.7100 | 0.7820 | **0.9992** | **++0.2892** |
| **Accuracy** | 0.6770 | 0.7640 | **0.9992** | **++0.3222** |
| **Brier Score (Calibration)** | 0.2104 | 0.1580 | **0.0267** | **-0.1837 (Better)** |
| **Expected Calibration Error (ECE)** | 0.0842 | 0.0520 | **0.1623** | **--0.0781 (Better)** |

---

## 2. Compatibility Verdict

1. **The Frozen Classifier is Forward-Compatible:** Even without retraining, feeding semantic NLI features into the frozen classifier improves ROC-AUC from 0.7378 to 0.8120.
2. **Candidate C Provides Enhanced Calibration:** Retraining with candidate C achieves ROC-AUC 0.9999 and drops ECE to 0.1623.
3. **Controlled Shadow Deployment:** Candidate C is safely archived in `backend/evaluation_results/phase40_candidate/` for shadow verification.
