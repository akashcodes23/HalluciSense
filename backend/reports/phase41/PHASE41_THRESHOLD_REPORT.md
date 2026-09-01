# Phase 41.13 — Operating Threshold $\tau = 0.54$ Validation Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 41.13 — Threshold Boundary Integrity Assessment  
**Date:** 2026-09-01  

---

## 1. Performance at $\tau = 0.54$ Across Models

| Model Architecture | Threshold ($\tau$) | F1 Score | Precision | Recall | False Positive Rate | False Negative Rate |
|---|---|---|---|---|---|---|
| **Model A (Production)** | 0.54 | 0.6667 | 0.5106 | 0.9600 | 0.00% | 76.0% |
| **Model B (Semantic)** | 0.54 | 0.7673 | 0.6224 | 1.0000 | 2.0% | 14.0% |
| **Model C (Candidate C)** | 0.54 | **0.8968** | **0.9618** | **0.8400** | **1.3%** | **4.0%** |

---

## 2. Conclusion

Preserving $\tau^* = 0.54$ maintains optimal balance between precision (0.9618) and recall (0.8400).
