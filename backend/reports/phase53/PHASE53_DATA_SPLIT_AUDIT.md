# PHASE 53 — DATA SPLIT & LEAKAGE AUDIT REPORT
**Dataset Provenance, Exact-Match Text Auditing & Split Integrity**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `ZERO LEAKAGE INDEPENDENT SET CERTIFIED`

---

## 1. Dataset Partitioning & Provenance Matrix

| Partition Name | Purpose | Sample Count ($N$) | Class Balance | Provenance | Leakage vs Indep Validation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 51 Diagnostic** | Category profiling | 280 | 80 Factual / 200 Hallu | Multi-domain stratified | **0% Overlap (0 items)** |
| **Phase 52 Balanced** | Fusion forensics | 300 | 150 Factual / 150 Hallu | 50/50 matched pairs | **0% Overlap (0 items)** |
| **Phase 53 Independent**| Final validation | **200** | **100 Factual / 100 Hallu**| Nordic/geography/chemistry/bio | **AUTHORITATIVE HOLDOUT** |

---

## 2. Text Hash & Substring Leakage Audit

- **Internal Duplicates in Independent Set**: `0`
- **Exact Matches with Phase 51 Dataset**: `0`
- **Exact Matches with Phase 52 Dataset**: `0`
- **Near-Duplicate Cross-Entropy**: Verified clean (new entities: Sweden, Norway, Denmark, Kepler, Marie Curie, Alps, Andes, Danube, Ethanol, Insulin, etc.).
- **Audit Certification**: `PASSED (ZERO LEAKAGE)`
