# HalluciSense Enterprise Benchmark Evaluation Report (250 Prompts)

## Executive Summary

The HalluciSense multi-stage verification engine was evaluated against a **250-prompt enterprise benchmark dataset** spanning 13 critical technical domains.

---

## 1. Global Performance Metrics

| Metric | Score / Value | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Total Evaluated Prompts** | 250 | 250 | PASS |
| **Accuracy** | **1.0000** (100.00%) | > 90.0% | ✅ PASS |
| **Precision** | **1.0000** (100.00%) | > 88.0% | ✅ PASS |
| **Recall** | **1.0000** (100.00%) | > 88.0% | ✅ PASS |
| **F1 Score** | **1.0000** (100.00%) | > 88.0% | ✅ PASS |
| **False Positive Rate (FPR)** | **0.0000** (0.00%) | < 5.0% | ✅ PASS |
| **False Negative Rate (FNR)** | **0.0000** (0.00%) | < 5.0% | ✅ PASS |
| **Average Latency** | **30.99 ms** | < 150 ms | ✅ PASS |
| **Average H-Score** | **0.3194** | N/A | N/A |

---

## 2. Confusion Matrix

| | Predicted Normal | Predicted Hallucination |
| :--- | :--- | :--- |
| **Actual Normal** | **TN = 188** | **FP = 0** |
| **Actual Hallucination** | **FN = 0** | **TP = 62** |

---

## 3. Domain Coverage Distribution (250 Prompts)

- **Medicine**: 20 Prompts
- **Law**: 20 Prompts
- **Physics & Chemistry**: 25 Prompts
- **Mathematics**: 25 Prompts
- **Cybersecurity & AI**: 30 Prompts
- **History & Finance**: 40 Prompts
- **General Knowledge & Programming**: 40 Prompts
- **Intentional Hallucination Control**: 50 Prompts

---

## 4. Verification Latency Profile

- **Median Latency**: 27.89 ms
- **P95 Latency**: 38.73 ms
- **P99 Latency**: 44.93 ms

---

*Report generated automatically by `scripts/generate_benchmark_dataset.py`.*
