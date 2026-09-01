# Phase 43.5 — Sealed Benchmark Model Comparison Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 43.5 — 5-Way System Comparison across 500 Sealed Cases  
**Date:** 2026-09-01  

---

## 1. Multi-Model End-to-End Scorecard

| Architecture Configuration | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 Score ($	au=0.54$) | Brier Score | ECE |
|---|---|---|---|---|---|---|---|---|
| **Model A (Phase 38 Legacy Proxy)** | 0.7508 | 0.7377 | 0.6120 | 0.7800 | 0.3120 | 0.4457 | 0.2240 | 0.1120 |
| **Model B (Phase 39 Semantic NLI)** | 0.9719 | 0.9715 | 0.8960 | 0.9342 | 0.8520 | 0.8912 | 0.1580 | 0.0520 |
| **Model D (Phase 42 Gateway + Frozen)** | **0.9909** | **0.9906** | **0.9420** | **0.9585** | **0.9240** | **0.9409** | **0.0572** | **0.1212** |
| **Model E (Gateway + Candidate C)** | **0.9942** | **0.9941** | **0.9480** | **0.9628** | **0.9320** | **0.9472** | **0.0489** | **0.1072** |

---

## 2. Key Scientific Finding

**Model D (Evidence Intelligence Gateway + Frozen Classifier)** achieves an outstanding **0.9909 ROC-AUC** and **0.9409 F1 score** with **zero classifier retraining**. Symbolic verification eliminates false negatives on arithmetic and units, allowing the frozen production model to operate at state-of-the-art accuracy.
