# HalluciSense Phase 26 Ablation Study Report

## System Component Contributions

| Ablation Variant | Accuracy | F1-Score | AUROC | $\Delta$ Acc |
|:---|:---:|:---:|:---:|:---:|
| **Full HalluciSense (Proposed)** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **Pillar 1 Only (Factual Grounding)** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **Pillar 2 Only (Confidence Engine)** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **Pillar 3 Only (Consistency Engine)** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **P1 + P2 Hybrid** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **P1 + P3 Hybrid** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **P2 + P3 Hybrid** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **w/o Adaptive Fusion (Fixed Weights)** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **w/o Calibration (Uncalibrated Raw)** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **w/o CrossEncoder (Dense Only)** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **w/o Retrieval (Zero Evidence)** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **w/o Token Localization** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
| **w/o Root Cause Classifier** | `1.0000` | `1.0000` | `1.0000` | `0.0000` |
