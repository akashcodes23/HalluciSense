# Phase 5.1 — Existing Reports Metric Provenance Audit

## Executive Summary

This audit independently inspects every metric reported across all previous HalluciSense engineering reports to classify each into its exact measurement provenance (**MEASURED**, **SIMULATED**, **ESTIMATED**, or **UNKNOWN**).

---

## 1. Metric Provenance Audit Matrix

| Metric Name | Current Reported Value | Evidence Source File | Measured? | Simulated? | Confidence Level | Auditor Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Single Gemini Call Guarantee | `llm_calls <= 1` | `tests/test_llm_budget.py` | YES | NO | HIGH | Maintain budget assertions in CI/CD pipeline. |
| Zero NaN Metric Rendering | `0 NaN displays` | `PillarCard.tsx & pipeline.py` | YES | NO | HIGH | Enforce safeScore formatting in React frontend. |
| Memory Growth Delta | `+0.03 MB RSS` | `scripts/run_1000_memory_leak_test.py` | YES | NO | HIGH | Monitor RSS growth in production container logs. |
| OWASP Security Audit | `100% Pass` | `scripts/run_security_audit.py` | YES | NO | HIGH | Rotate API keys every 90 days. |
| 500-Prompt Benchmark Accuracy | `100.0%` | `scripts/evaluate_500_benchmark.py` | NO | YES | MEDIUM | Classify as Synthetic Evaluation Benchmark. |
| 500-User Stress Throughput | `11,964 RPS` | `scripts/run_stress_test.py` | NO | YES | MEDIUM | Classify as Simulated Concurrency Benchmark. |

---

*Report generated automatically by `scripts/run_phase5_master_audit.py`.*
