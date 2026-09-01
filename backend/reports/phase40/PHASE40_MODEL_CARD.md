# Phase 40.19 — HalluciSense Candidate Model Card

**Model Identifier:** `hybrid_meta_classifier_phase40_candidate` (Version `phase40_candidate_v1`)  
**Base Architecture:** `sklearn.ensemble.HistGradientBoostingClassifier` (19 features, `RobustScaler`)  
**Target Class:** 0 = Factual, 1 = Hallucinated  
**Operating Threshold:** $\tau^* = 0.54$  
**Evaluation Date:** 2026-09-01  

---

## 1. Model Summary

- **ROC-AUC:** **0.9999**
- **PR-AUC:** **0.9998**
- **F1 Score:** **0.9992**
- **Expected Calibration Error:** **0.1623**
- **Brier Score:** **0.0267**
- **Training Samples ($N$):** 40,601 (70% split of 58,002 clean records)
- **Validation Samples ($N$):** 8,700 (15% split)
- **Test Samples ($N$):** 8,701 (15% split)

---

## 2. Intended Use & Safety Bounds

- **Intended Use:** Detection of factual hallucinations, contradictions, and ungrounded statements in LLM outputs.
- **Scientific Caveat:** Explanations represent local counterfactual attributions against training medians. They are not causal proofs.
