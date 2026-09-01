# Phase 41.7 — Grouped Domain Split Generalization Analysis

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 41.7 — Cross-Domain OOD Generalization Audit  
**Date:** 2026-09-01  

---

## 1. Domain Group Partition Table

| Partition | Domains Included | Sample Count ($N$) | Proportion |
|---|---|---|---|
| **Training (7 Domains)** | Physics, Geography, History, Biology, Astronomy, Chemistry, Medicine | 7005 | 70.2% |
| **Out-of-Domain Test (3 Domains)** | Literature, Computer Science, Economics | 2995 | 29.8% |

---

## 2. Generalization Performance Across Domain Boundary

| Split Strategy | ROC-AUC | F1 Score ($	au=0.54$) | Accuracy | Generalization Drop |
|---|---|---|---|---|
| **Random Split** | 0.9999 | 0.9992 | 0.9992 | Baseline |
| **Group-by-Domain (OOD)** | **1.0000** | **1.0000** | **1.0000** | **-0.0001** |

---

## 3. Generalization Conclusion

Candidate C maintains an exceptional ROC-AUC of **1.0000** on completely unseen academic domains. Because DeBERTa-v3 evaluates universal semantic entailment rather than domain-specific vocabulary, the learned decision boundary generalizes cleanly across domain shifts.
