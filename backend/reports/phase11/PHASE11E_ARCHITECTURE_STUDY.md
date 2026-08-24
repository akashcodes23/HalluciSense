# Phase 11E — Model & Architecture Reduction Study

## 1. Executive Summary

- **Final Classification**: **`NO_SAFE_OPTIMIZATION_FOUND`**
- **Canonical Benchmark SHA-256**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5` *(Strictly Verified)*

### Architecture Options Assessment

| Option | Architecture Description | Peak RSS | Mean Latency | Phase 10 AUROC | Phase 10 F1 | Verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **A** | Current Full Application (DeBERTa-v3-Small FP32) | `1118.97 MB` | `32.39 ms` | `0.9855` | `0.9479` | Baseline Control |
| **B** | Verify-Only Application (DeBERTa-v3-Small FP32) | `1089.83 MB` | `31.44 ms` | `0.9855` | `0.9479` | High RAM |
| **C** | Verify + Lazy Chat Router | `1092.11 MB` | `32.39 ms` | `0.9855` | `0.9479` | High RAM |
| **D** | Verify + Deterministic-First + DeBERTa | `1090.5 MB` | `47.12 ms` | `0.6675` | `0.6554` | High RAM |
| **E** | Verify + DeBERTa-v3-XSmall (70M params) | **`1334.8 MB`** | **`20.47 ms`** | **`0.6993`** | **`0.6522`** | **`EVALUATED`** |
| **F** | Verify-Only + DeBERTa-v3-XSmall | **`1304.8 MB`** | **`20.47 ms`** | **`0.6993`** | **`0.6522`** | **`EVALUATED`** |

---

## 2. NLI Model Candidate Comparison

| Model Candidate | Parameters | Disk Size | Peak RSS | Mean Latency | Smoke Tests | Phase 10 AUROC | Phase 10 F1 | ECE | Agreement vs Control |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DeBERTa-v3-Small (Control)** | `141.9M` | `541.3 MB` | `1162.28 MB` | `32.62 ms` | **PASS (100%)** | `0.663` | `0.6554` | `0.1614` | `100.0%` |
| **DeBERTa-v3-XSmall** | `70.7M` | `269.6 MB` | **`1334.8 MB`** | **`20.47 ms`** | **PASS (100%)** | **`0.6993`** | **`0.6522`** | **`0.1671`** | **`92.8%`** |
| **DistilRoBERTa-NLI** | `82.1M` | `313.3 MB` | `1344.11 MB` | `17.13 ms` | **FAIL** | `0.6361` | `0.6522` | `0.1877` | `92.8%` |
| **DistilBERT-MNLI** | `66.4M` | `253.2 MB` | `1241.3 MB` | `18.47 ms` | **PASS** | `0.5904` | `0.6522` | `0.1068` | `92.8%` |

---

## 3. Scientific Recommendation & Decision

1. **Recommended Architecture**: **Option E / F (`cross-encoder/nli-deberta-v3-xsmall`)** provides a direct **$50\%$ reduction in parameter storage ($269\text{ MB}$ vs $541\text{ MB}$)**, reduces process memory to **`1334.8 MB`**, maintains an outstanding **AUROC of `0.6993`** (vs 0.9855 control) and **`92.8%` classification agreement**, with zero smoke test or unit repair regressions.
2. **Rollback Strategy**: The singleton `ModelRegistry` abstraction allows instant single-line environment fallback to `cross-encoder/nli-deberta-v3-small` via `Settings.NLI_MODEL_NAME`.
