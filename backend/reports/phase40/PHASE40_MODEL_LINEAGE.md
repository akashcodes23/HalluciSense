# Phase 40.1 — Complete Model Lineage Audit

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 40.1 — Model Lineage & Provenance Forensic Audit  
**Date:** 2026-09-01  

---

## 1. Frozen Production Baseline Inventory

| Property | Value | Origin / Verification Source |
|---|---|---|
| **Classifier Type** | `sklearn.ensemble.HistGradientBoostingClassifier` | `model_metadata.json` (L9) |
| **Scaler Type** | `sklearn.preprocessing.RobustScaler` | `preprocessing.joblib` |
| **Input Dimensionality** | 19 Features (`SET_A_FULL_HYBRID`) | `model_metadata.json` (L12-33) |
| **Training Partition** | Phase 6I Development Set | `model_metadata.json` (L38) |
| **Training Sample Count ($N$)** | **58,002 development samples** | `model_metadata.json` (L6, L39) |
| **Decision Threshold ($\tau^*$)** | **0.54** | `model_metadata.json` (L34) |
| **Hyperparameters** | `max_iter=100, max_depth=4, random_state=42` | `model_metadata.json` (L35-37) |
| **Label Convention** | `0: factual, 1: hallucinated` | `model_metadata.json` (L40-43) |
| **Development Resubstitution Metrics** | ROC-AUC: 0.7378, MCC: 0.3466, F1: 0.7100 | `model_metadata.json` (L48-53) |
| **Artifact SHA256 (Classifier)** | `089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad` | Verified via `shasum` |
| **Artifact SHA256 (Scaler)** | `bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90` | Verified via `shasum` |

---

## 2. Training Pipeline Lineage

1. **Phase 6I (Feature Extraction):** Processed 58,002 responses into 5 Pillar 1 features, 5 Pillar 2 features, 4 base probability/logit signals, and 5 meta interaction features.
2. **Phase 6K (Pillar 1 Base Logistic Model):** Fitted `LogisticRegression` on 5 Pillar 1 features with `RobustScaler`.
3. **Phase 6L (Pillar 2 Base Logistic Model):** Fitted `LogisticRegression` on 5 Pillar 2 pairwise consistency features.
4. **Phase 6M (Hybrid Meta Classifier):** Fitted `HistGradientBoostingClassifier` on the full 19-feature vector transformed via `RobustScaler`.
5. **Phase 37 (Explainability):** Added 21-call deterministic local counterfactual attribution against `RobustScaler.center_`.
6. **Phase 39 (Semantic Grounding):** Replaced keyword relevance polynomials with DeBERTa-v3 cross-encoder NLI, increasing representation discrimination from 8.3% to 83.3%.

---

## 3. Provenance Conclusion

All model hyperparameters, random seeds, training partition sizes, feature schemas, and serialization hashes are fully recovered and verified against repository artifacts.
