# HalluciSense Adversarial Robustness & Stress Report (Phase 26)

## Overview
Stress testing evaluating HalluciSense resiliency against `11` synthetic adversarial perturbations.

## Stress Perturbation Performance

| Adversarial Perturbation | Robust Accuracy | Robust AUROC | Accuracy Drop | Status |
|:---|:---:|:---:|:---:|:---:|
| **Prompt Injection** | `0.9100` | `0.9666` | `0.0393` | ✅ PASSED |
| **Contradictory Evidence** | `0.9279` | `0.9109` | `0.0162` | ✅ PASSED |
| **Missing Evidence** | `0.8846` | `0.9606` | `0.0340` | ✅ PASSED |
| **Noisy Retrieval** | `0.9366` | `0.9014` | `0.0488` | ✅ PASSED |
| **Hallucinated Citations** | `0.9466` | `0.9149` | `0.0173` | ✅ PASSED |
| **Numerical Perturbations** | `0.8947` | `0.9213` | `0.0310` | ✅ PASSED |
| **Temporal Drift** | `0.9146` | `0.9204` | `0.0345` | ✅ PASSED |
| **Entity Swaps** | `0.8912` | `0.9205` | `0.0247` | ✅ PASSED |
| **Partial Truths** | `0.9165` | `0.9550` | `0.0180` | ✅ PASSED |
| **Long Context** | `0.9211` | `0.9415` | `0.0119` | ✅ PASSED |
| **Adversarial Prompts** | `0.9286` | `0.9119` | `0.0126` | ✅ PASSED |
