# Phase 6E: Experimental Freeze Document

**Date**: 2026-08-11  
**Git SHA**: `79c92e6fa5191ba2e3df89034ff210e26826b1bd`  
**Status**: ARCHITECTURE, WEIGHTS, AND THRESHOLDS FROZEN  

---

## 1. Frozen System Configuration

### Production Fusion Weights
- $\alpha = 0.40$ (Pillar 1: Retrieval + NLI Verification)
- $\beta = 0.30$ (Pillar 2: Self-Consistency / Uncertainty)
- $\gamma = 0.30$ (Pillar 3: Modality & Temporal Verification)
- Constraint: $\alpha + \beta + \gamma = 1.00$ (**FROZEN**)

### Production Risk Thresholds
- `VERIFIED`: $< 0.35$
- `NEEDS_VERIFICATION`: $< 0.50$
- `MODERATE_RISK`: $< 0.65$
- `LIKELY_HALLUCINATED`: $\ge 0.65$ (**FROZEN**)

### Epistemic & Temporal Engine
- NLI Model: `cross-encoder/nli-deberta-v3-small`
- Epistemic Resolver: `EpistemicResolver` (`app/core/engine/epistemic.py`)
- Temporal Claim Engine: `TemporalClaimEngine` (`app/core/engine/temporal.py`)
- Date Resolver: `EventTemporalAnchorResolver` (`app/core/engine/pillar1_retrieval.py`)

---

## 2. Environment Execution Manifest

| Environment Key | Value |
|:---|:---|
| **Git SHA** | `79c92e6fa5191ba2e3df89034ff210e26826b1bd` |
| **Python Version** | `3.10.12` |
| **PyTorch Version** | `2.5.1` |
| **Transformers Version** | `4.47.1` |
| **OS** | macOS |
| **Random Seed** | `42` (Fixed for evaluation and bootstrap resampling) |
| **NLI Model** | `cross-encoder/nli-deberta-v3-small` |
