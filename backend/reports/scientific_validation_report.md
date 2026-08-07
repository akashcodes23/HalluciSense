# HalluciSense Phase 25 Scientific Validation Report

**Experiment ID**: `EXP_PHASE25_7AF68AF5`  
**Timestamp**: `2026-08-06 11:09:56 UTC`  
**Execution Runtime**: `3926.67s`  

---

## 1. Executive Performance Metrics

| Evaluation Suite | Sample Size | Empirical Accuracy | Target Gate | Status |
|:---|:---:|:---:|:---:|:---:|
| **Regression Suite v2** | `1000` | **`88.30%`** | $\ge 90.0\%$ | ❌ FAILED |
| **Long-Form Scientific QA** | `500` | **`52.20%`** | $\ge 85.0\%$ | ❌ FAILED |
| **Retrieval Recall@5** | `20` | **`0.6000`** | $\ge 0.85$ | ❌ FAILED |
| **Calibration ECE** | - | **`0.0245`** | $\le 0.08$ | ✅ PASSED |

---

## 2. Information Retrieval (IR) Diagnostics

- **Recall@1**: `0.6000`
- **Recall@5**: `0.6000`
- **Mean Reciprocal Rank (MRR)**: `0.6000`
- **nDCG@5**: `1.0000`
- **Evidence Coverage**: `1.0000`

---

## 3. Domain-Wise Accuracy Breakdown (Long-Form QA)

| Domain | Evaluated Samples | Accuracy |
|:---|:---:|:---:|
| **Medicine** | `50` | **`34.0%`** |
| **Physics** | `50` | **`68.0%`** |
| **Biology** | `50` | **`34.0%`** |
| **Chemistry** | `50` | **`34.0%`** |
| **Finance** | `50` | **`50.0%`** |
| **Law** | `50` | **`50.0%`** |
| **Programming** | `50` | **`50.0%`** |
| **History** | `50` | **`100.0%`** |
| **Education** | `50` | **`50.0%`** |
| **Scientific QA** | `50` | **`52.0%`** |

---

## 4. Root-Cause Taxonomy Failure Distribution

| Failure Category | Sample Count | Percentage |
|:---|:---:|:---:|
| **Retrieval Failure** | `406` | `40.6%` |
| **VERIFIED** | `391` | `39.1%` |
| **Entity Linking Failure** | `195` | `19.5%` |
| **Evidence Missing** | `8` | `0.8%` |

---

## 5. Artifact Verification & Figure Package
- Generated **`9`** 600 DPI publication figures in SVG, PDF, and PNG in `reports/figures/`.
- Full stage execution traces saved to `backend/traces/`.
