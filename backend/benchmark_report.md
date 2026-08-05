# HalluciSense Benchmark Evaluation Report (500 Prompts)

> **Notice**: Evaluation Status: **Synthetic Evaluation Suite (Offline Validation Benchmark)**.

---

## 1. Global Performance Metrics

| Metric | Measured Score | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Evaluated Prompts** | **500** | 500 | ✅ PASS |
| **Accuracy** | **1.0000** (100.00%) | > 90.0% | ✅ PASS |
| **Precision** | **1.0000** (100.00%) | > 88.0% | ✅ PASS |
| **Recall** | **1.0000** (100.00%) | > 88.0% | ✅ PASS |
| **F1 Score** | **1.0000** (100.00%) | > 88.0% | ✅ PASS |
| **False Positive Rate (FPR)** | **0.0000** (0.00%) | < 5.0% | ✅ PASS |
| **False Negative Rate (FNR)** | **0.0000** (0.00%) | < 5.0% | ✅ PASS |
| **Average Latency** | **25.03 ms** | < 150 ms | ✅ PASS |
| **Average H-Score** | **0.5997** | N/A | N/A |

---

## 2. Confusion Matrix

| | Predicted Normal | Predicted Hallucination |
| :--- | :--- | :--- |
| **Actual Normal** | **TN = 166** | **FP = 0** |
| **Actual Hallucination** | **FN = 0** | **TP = 334** |

---

## 3. Latency Distribution Histogram

- **P50 Latency**: 23.03 ms
- **P95 Latency**: 32.04 ms
- **P99 Latency**: 37.04 ms

---

*Report generated automatically by `scripts/evaluate_500_benchmark.py`.*
