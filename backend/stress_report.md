# Sprint 5 — HalluciSense Concurrency & Stress Testing Report

> **Notice**: Evaluation Status: **Provider Invocations Mocked / Simulated Runtime**.

---

## 1. Concurrency Benchmarks (10 – 500 Virtual Users)

| Virtual Users | Throughput (RPS) | Avg Latency | P50 Latency | P95 Latency | P99 Latency | WebSocket Errors | Quota Trips | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 218.01 | 23.39 ms | 22.36 ms | 37.01 ms | 37.13 ms | 0 | 0 | PASS |
| 25 | 664.77 | 26.61 ms | 28.87 ms | 36.73 ms | 37.27 ms | 0 | 0 | PASS |
| 50 | 1388.12 | 21.98 ms | 22.17 ms | 33.45 ms | 35.14 ms | 0 | 0 | PASS |
| 100 | 2592.87 | 25.2 ms | 25.04 ms | 36.38 ms | 37.85 ms | 0 | 0 | PASS |
| 250 | 6251.99 | 23.74 ms | 22.66 ms | 36.17 ms | 37.56 ms | 0 | 0 | PASS |
| 500 | 11964.04 | 24.72 ms | 24.58 ms | 36.27 ms | 37.82 ms | 0 | 0 | PASS |

---

## 2. Infrastructure Resource Profile (500 Virtual Users)

- **CPU Utilization**: 28.4% peak across worker pool.
- **RAM Usage**: 412 MB RSS.
- **Upstash Redis Connection Pool**: 14 / 100 active connections.
- **PostgreSQL Pool**: 18 / 50 active connections.
- **WebSocket Failures**: 0 (100% connection stability).
- **Timeouts**: 0 (0% dropped frames).

---

## 3. Bottlenecks & Operational Recommendations

1. **Worker Pool Sizing**: Maintain 4 Uvicorn async worker processes for high-throughput scaling (> 10,000 RPS).
2. **Database Pooling**: Configure `max_overflow=20` on asyncpg SQLAlchemy engine pool.

---

*Report generated automatically by `scripts/run_stress_test.py`.*
