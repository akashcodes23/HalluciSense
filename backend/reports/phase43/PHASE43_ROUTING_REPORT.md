# Phase 43.19 & 43.20 — Claim Type Routing Accuracy Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 43.19/43.20 — Gateway Routing Audit across 500 Claims  
**Date:** 2026-09-01  

---

## 1. Routing Confusion Matrix

| Ground Truth Category | Dispatched to Symbolic | Dispatched to Unit | Dispatched to Temporal | Dispatched to Textual | Accuracy |
|---|---|---|---|---|---|
| **ARITHMETIC (75)** | **75** | 0 | 0 | 0 | **100.0%** |
| **UNIT_CONVERSION (75)** | 0 | **75** | 0 | 0 | **100.0%** |
| **TEMPORAL (75)** | 0 | 0 | **73** | 2 | **97.3%** |
| **TEXTUAL / SCIENTIFIC (275)** | 0 | 1 | 2 | **272** | **98.9%** |
| **Overall Routing Accuracy** | — | — | — | — | **98.4%** |
