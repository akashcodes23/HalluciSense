# PHASE 50 — SYSTEM BASELINE & ENVIRONMENT REPORT
**Zero-OOM Production Forensics & Memory Profile**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `BASELINE AUDITED`

---

## 1. System & Dependency Environment

| Parameter | Measured Specification |
| :--- | :--- |
| **Git Baseline SHA** | `5689cc0` |
| **Python Version** | `3.10.12 (CPython x86_64)` |
| **PyTorch Version** | `2.6.0 (CPU-only build)` |
| **Transformers Version** | `4.49.0` |
| **NumPy Version** | `2.2.3` |
| **FastAPI / Uvicorn** | `FastAPI 0.115.8 / Uvicorn 0.34.0` |
| **OS / Environment** | `macOS 15.3 / Darwin 24.3.0` |
| **Configured Uvicorn Workers** | `1 (Strictly 1)` |
| **Configured OpenMP / MKL Threads** | `OMP=1, MKL=1, OPENBLAS=1, NUMEXPR=1` |
| **Memory Ceiling (Railway)** | `1024 MB` |

---

## 2. Invariant Model Checksums & Signatures

| Artifact | Path | SHA256 Hash | Status |
| :--- | :--- | :--- | :--- |
| **Hybrid Meta Classifier** | `backend/app/models/hybrid_meta_classifier.joblib` | `089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad` | `FROZEN & IMMUTABLE` |
| **Feature Scaler** | `backend/app/models/preprocessing.joblib` | `bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90` | `FROZEN & IMMUTABLE` |
| **Decision Threshold** | `tau = 0.54` | Exact scalar constant | `IMMUTABLE` |
| **Feature Schema** | 19 canonical features | Exact vector schema | `IMMUTABLE` |
