# Phase 11E — Memory Breakdown & Component Analysis

## 1. Clean Subprocess Component Memory Audit

| Component / Layer | Process RSS | Net Delta | Time |
|---|:---:|:---:|:---:|
| **Base Python Process** | `15.41 MB` | `+0.0 MB` | `0.0 ms` |
| **FastAPI Framework** | `60.55 MB` | `+45.16 MB` | `222.56 ms` |
| **PyTorch CPU Core** | `170.45 MB` | `+155.06 MB` | `578.49 ms` |
| **Transformers Library** | `184.31 MB` | `+169.11 MB` | `692.42 ms` |
| **DeBERTa Tokenizer** | `480.97 MB` | `+465.53 MB` | `2237.24 ms` |
| **DeBERTa Model (FP32)** | `692.0 MB` | `+676.78 MB` | `3975.01 ms` |
| **Correction Engine** | `908.03 MB` | `+892.62 MB` | `5354.83 ms` |

---

## 2. Chat vs Verify Profile Separation

| Execution Profile | Peak RSS | Elapsed Time |
|---|:---:|:---:|
