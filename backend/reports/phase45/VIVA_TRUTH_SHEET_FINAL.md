# HalluciSense — Final Viva Truth Sheet

**Repository:** akashcodes23/HalluciSense  
**Authors:** Final Year Engineering Project Examination Committee  
**Date:** 2026-09-01  

---

## 1. Verified Scientific Facts (Supported by Repository Artifacts)

| Scientific Metric | Exact Value | Source Dataset & Artifact | Evidence / Methodology |
|---|---|---|---|
| **Training Dataset Size** | **58,002** samples | `dataset_58k_clean.parquet` | Stratified clean partition |
| **Feature Dimensionality** | **19** features | `SET_A_FULL_HYBRID` in `pipeline.py` | 5 P1 + 10 P2 + 4 Meta features |
| **Operating Threshold** | $\tau^* = \mathbf{0.54}$ | `production_model_manifest.json` | Optimal Youden J validation point |
| **Production Model Hash** | `089ebd2d277d1c21...` | `hybrid_meta_classifier.joblib` | SHA256 verified |
| **Label-Shuffle Sanity AUC** | **0.4974** ($pprox 0.50$) | `PHASE41_RANDOMIZATION_RESULTS.md` | Proves zero label/dataset leakage |
| **Sealed Red-Team ROC-AUC** | **0.9978** | `PHASE45_PERFORMANCE.md` | 500-case sealed holdout benchmark |
| **Sealed Red-Team F1 Score** | **0.9721** | `PHASE45_PERFORMANCE.md` | $\tau = 0.54$ operating point |
| **Minimal-Pair Discrimination** | **90.0%** separation | `PHASE43_FINAL_REPORT.md` | Increased from 8.3% (Phase 38) |
| **Representation Collapse** | **16.7%** | `PHASE39_FINAL_REPORT.md` | Reduced from 91.7% |
| **Production Peak RSS** | **~539.8 MB** | `PHASE44_OBSERVABILITY.md` | 484 MB free under 1024 MB limit |
| **Exit 137 / OOM Crashes** | **0** | Railway runtime logs | 1 worker, single NLI singleton |
