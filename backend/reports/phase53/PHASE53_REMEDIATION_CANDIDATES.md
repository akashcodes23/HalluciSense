# PHASE 53 — REMEDIATION CANDIDATES BENCHMARKING REPORT
**Repeated 5x5 Stratified Cross-Validation on Development Set ($N=300$)**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `DEVELOPMENT CANDIDATES BENCHMARKED`

---

## 1. Repeated 5x5 Stratified CV Results Matrix (Mean $\pm$ Std)

| Candidate Architecture | AUROC | AUPRC | Accuracy | Recall | Specificity | F1 | MCC | Brier | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate B: Regularized HGBoost** | **0.8528 $\pm$ 0.04** | **0.8552 $\pm$ 0.05** | **76.13% $\pm$ 4.0** | **72.93% $\pm$ 9.6** | **79.33% $\pm$ 9.6** | **0.7513 $\pm$ 0.05** | **0.5317 $\pm$ 0.08** | **0.1590** | **0.1275** |
| **Candidate D: HGBoost (Selected Subset)**| **0.8608 $\pm$ 0.04** | **0.8629 $\pm$ 0.05** | **76.60% $\pm$ 4.1** | **73.87% $\pm$ 10.7**| **79.33% $\pm$ 10.1**| **0.7565 $\pm$ 0.05** | **0.5422 $\pm$ 0.08** | **0.1541** | **0.1196** |
| **Candidate A: Calibrated Logistic Reg** | 0.8410 $\pm$ 0.04 | 0.8523 $\pm$ 0.04 | 74.53% $\pm$ 5.5 | 71.20% $\pm$ 8.6 | 77.87% $\pm$ 8.5 | 0.7351 $\pm$ 0.06 | 0.4958 $\pm$ 0.11 | 0.1679 | 0.1294 |
| **Candidate C: Random Forest** | 0.8397 $\pm$ 0.05 | 0.8469 $\pm$ 0.06 | 73.67% $\pm$ 5.9 | 66.53% $\pm$ 10.0| 0.8080 $\pm$ 0.08 | 0.7137 $\pm$ 0.07 | 0.4827 $\pm$ 0.12 | 0.1653 | 0.1287 |
| **Candidate E: Monotonic Logistic Reg** | 0.8394 $\pm$ 0.04 | 0.8505 $\pm$ 0.04 | 73.47% $\pm$ 5.6 | 70.80% $\pm$ 7.9 | 76.13% $\pm$ 8.4 | 0.7265 $\pm$ 0.06 | 0.4732 $\pm$ 0.11 | 0.1698 | 0.1291 |
| **Frozen Production Baseline ($\tau=0.54$)**| 0.6905 | 0.5883 | 57.00% | 30.67% | 83.33% | 0.4163 | 0.1647 | 0.2516 | 0.2043 |

---

## 2. Selected Primary Candidate: Candidate B (HistGradientBoosting)

- **Architecture Choice**: `HistGradientBoostingClassifier` with $L_2$ regularization ($\lambda = 1.5$), shallow tree depth ($\text{max\_depth} = 3$), and min leaf samples ($15$).
- **Performance**: High balanced accuracy ($76.13\%$), high recall ($72.93\%$), high specificity ($79.33\%$), and MCC exceeding **0.53**.
- **Candidate Artifact Serialization**:
  * Model: `backend/evaluation_results/phase53/candidate/hybrid_meta_classifier_phase53_candidate.joblib`
  * Scaler: `backend/evaluation_results/phase53/candidate/preprocessing_phase53_candidate.joblib`
  * Schema: `backend/evaluation_results/phase53/candidate/candidate_schema.json`
  * Metadata: `backend/evaluation_results/phase53/candidate/candidate_metadata.json`
