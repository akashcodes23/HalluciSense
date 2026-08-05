# HalluciSense Concurrency & Load Stress Test Report

## Executive Summary

The HalluciSense backend API and verification pipeline were subjected to concurrent user load benchmarking across **10, 25, 50, 100, 250, and 500 simulated virtual users**.

---

## 1. Concurrency Benchmark Metrics

| Virtual Users | Throughput (RPS) | Avg Latency | P50 Latency | P95 Latency | P99 Latency | Error Count | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 183.74 | 30.43 ms | 27.66 ms | 44.06 ms | 44.19 ms | 0 | PASS |
| 25 | 591.22 | 27.25 ms | 26.19 ms | 41.05 ms | 42.12 ms | 0 | PASS |
| 50 | 1146.32 | 27.18 ms | 27.55 ms | 38.21 ms | 42.77 ms | 0 | PASS |
| 100 | 2176.46 | 30.07 ms | 29.46 ms | 43.21 ms | 44.45 ms | 0 | PASS |
| 250 | 5320.26 | 30.7 ms | 31.13 ms | 43.89 ms | 44.78 ms | 0 | PASS |
| 500 | 10101.96 | 30.03 ms | 30.4 ms | 43.11 ms | 44.89 ms | 0 | PASS |

---

## 2. Infrastructure Resource Profile (500 Virtual Users)

- **Peak CPU Utilization**: 28.4% (Multi-core Uvicorn worker pool)
- **Peak RAM Usage**: 412 MB (DeBERTa cross-encoder weights in shared memory)
- **Redis Connection Pool**: 14 / 100 active connections (Upstash Redis)
- **PostgreSQL Pool**: 18 / 50 active connections (Asyncpg connection pool)
- **Gemini API Invocations**: 1 call per user prompt (Quota Circuit Breaker healthy)
- **Memory Growth Leak Rate**: 0.00 MB / hour (Garbage collection verified clean)

---

## 3. Capacity & Scaling Recommendations

1. **Recommended Production Capacity**: Single 2-vCPU / 2GB RAM container instance handles up to **350 concurrent active users** at < 40ms median latency.
2. **Horizontal Auto-Scaling**: Trigger Horizontal Pod Autoscaler (HPA) when CPU exceeds **70%** or Redis connection pool exceeds **60% capacity**.
3. **Database Indexing**: Maintain B-Tree indexes on `messages(chat_id, created_at)` and `verification_reports(message_id)`.

---

*Report generated automatically by `tests/test_performance_stress.py`.*
