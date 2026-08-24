# Phase 11D — Optimized NLI Runtime Evaluation

## 1. Executive Summary & Runtime Evaluation Decision

**Final Classification**: **`ONNX_REJECTED`**

| Metric | PyTorch FP32 (Control) | ONNX FP32 | ONNX Dynamic INT8 |
|---|:---:|:---:|:---:|
| **Disk Model Size** | `541.29 MB` | `541.76 MB` | `164.32 MB` |
| **Startup RSS** | `35.83 MB` | `36.12 MB` | `35.7 MB` |
| **Model Load RSS** | `844.89 MB` | `1883.22 MB` | `946.58 MB` |
| **First Inference RSS** | `1026.41 MB` | `1885.25 MB` | `947.55 MB` |
| **Peak RSS** | **`1118.97 MB`** | **`1891.19 MB`** | **`1326.34 MB`** |
| **Sequential Mean Latency** | `16.9 ms` | `31.44 ms` | `11.91 ms` |
| **Sequential P95 Latency** | `19.05 ms` | `42.63 ms` | `15.77 ms` |
| **Sequential Errors** | `0` | `0` | `0` |
| **Concurrent Mean Latency** | `42.7 ms` | `35.2 ms` | `13.14 ms` |
| **Concurrent Errors** | `0` | `0` | `0` |
| **Scientific Agreement vs Control** | `100.0%` (Self) | **`100.0%`** | **`100.0%`** |
| **H-Score Mean Absolute Error (MAE)** | `0.0000` | **`0.0`** | **`0.2194`** |
| **Max H-Score Delta** | `0.0000` | **`0.0`** | **`0.4246`** |
| **Smoke Test Suite** | **PASS (100%)** | **PASS (100%)** | **PASS (100%)** |
| **Regression Test Suite** | **PASS (76/76)** | **PASS (76/76)** | **PASS (76/76)** |

---

## 2. Numerical Equivalence (ONNX vs PyTorch)

- **ONNX FP32 Max Logit Difference**: `1.4e-05`
- **ONNX FP32 MAE**: `4e-06`
- **ONNX Dynamic INT8 Max Logit Difference**: `1.842161`
- **ONNX Dynamic INT8 MAE**: `0.769162`

---

## 3. Scientific Smoke Cases Verification

1. **True Speed of Light** ($299,792,458\text{ m/s}$): **VERIFIED** across all runtimes ($H \le 0.35$).
2. **False Speed of Light** ($299,792,458\text{ km/s}$): **LIKELY_HALLUCINATED** across all runtimes ($H \ge 0.65$).
3. **True Water** ($H_2O$): **VERIFIED** across all runtimes ($H \le 0.35$).
4. **False Water** ($CO_2$): **LIKELY_HALLUCINATED** across all runtimes ($H \ge 0.65$).
5. **Negation Inversion** (*Mitochondria do not produce ATP*): **LIKELY_HALLUCINATED** across all runtimes ($H \ge 0.65$).
6. **Closed-Loop Numerical Repair**: $299792458\text{ km/s} \to 299792458\text{ m/s}$ followed by re-verification **PASS**.

---

## 4. Benchmark Invariant Audit

- **Canonical Dataset SHA-256**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5` *(Strictly Verified)*
- **Sample Evaluated**: 50 representative benchmark pairs.
