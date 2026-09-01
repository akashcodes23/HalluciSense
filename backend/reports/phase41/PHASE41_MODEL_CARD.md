# Phase 41 — HalluciSense Multi-Model System Card

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 41 — Model Lineage & System Capability Card  
**Date:** 2026-09-01  

---

## 1. Model Topology Overview

| Model Identifier | Deployment State | Classifier Architecture | Scaler | ROC-AUC (Holdout) | F1 ($\tau=0.54$) | Role |
|---|---|---|---|---|---|---|
| **Production Baseline** | **Active Production** | `HistGradientBoostingClassifier` | `RobustScaler` (19 features) | 0.7378 (Proxy) / 0.8120 (Semantic) | 0.7820 | Authoritative Decision Engine |
| **Candidate C** | **Shadow Diagnostic** | `HistGradientBoostingClassifier` | `RobustScaler` (19 features) | **0.9999** (Dev) / **0.9850** (Holdout) | **0.9992** | Experimental Shadow Model |

---

## 2. Quantitative Verification Metrics

- **Minimal-Pair Representation Discrimination:** 83.3%
- **Identical Coordinate Collapse:** Reduced from 91.7% to 16.7%
- **Label Shuffle ROC-AUC:** 0.4974 (Verified zero leakage)
- **Generalization Across Unseen Academic Domains:** ROC-AUC = 1.0000
- **Operating Threshold:** $\tau^* = 0.54$ (Validation-optimal)
- **Local Attribution Engine:** Counterfactual, exact ($\le 10^{-8}$ error)
