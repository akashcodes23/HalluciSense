# PHASE 52 — BASELINE & FORENSIC DISCREPANCY AUDIT
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `EMPIRICAL FORENSIC BASELINE`

---

## 1. Production Invariants & Baseline Checksums

- **Current Production Baseline Commit**: `f591b11` (Phase 51)
- **Frozen Classifier SHA256**: `089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad` (`hybrid_meta_classifier.joblib`)
- **Frozen Scaler SHA256**: `bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90` (`preprocessing.joblib`)
- **Frozen Decision Threshold $\tau^*$**: `0.54`
- **Canonical Schema**: `19 features` (SET_A_FULL_HYBRID)

---

## 2. Phase 51 Metrics & The Primary Forensic Anomaly

In Phase 51, full diagnostic evaluation on $N=280$ stratified samples revealed:
- **P1 Grounding Alone AUROC**: **0.8341**
- **Full Frozen Detector AUROC**: **0.7183**
- **Frozen Operating Point**: Accuracy: **46.79%**, Precision: **84.93%**, Recall: **31.00%**, Specificity: **86.25%**, MCC: **0.1775**, F1: **0.4542**.

### The Primary Forensic Anomaly:
$$\text{AUROC}(\text{P1 Grounding}) = 0.8341 \gg \text{AUROC}(\text{Full Hybrid Detector}) = 0.7183$$

The upstream evidence grounding subsystem generates a strong, discriminative factual signal ($\text{AUROC} > 0.83$), but the downstream 19-feature vector assembly, preprocessing scaler, and frozen `HistGradientBoostingClassifier` tree splits degrade this signal by over 11.5 AUROC points, resulting in an overwhelming false negative rate ($FN = 138 / 200$) and low recall ($31.00\%$).
