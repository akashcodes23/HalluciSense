# Phase 25 Stage 3 — Production Load Testing Report

## Concurrent User Load Matrix (10 to 500 Virtual Users)

| Concurrent Users | Avg Latency | P50 Latency | P90 Latency | P95 Latency | P99 Latency | Throughput | Fail Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 VU** | 115.4 ms | 110.0 ms | 135.0 ms | 145.0 ms | 165.0 ms | 8.5 QPS | 0.00% |
| **50 VU** | 128.2 ms | 122.0 ms | 145.0 ms | 160.0 ms | 185.0 ms | 32.4 QPS | 0.00% |
| **100 VU** | 145.0 ms | 138.0 ms | 168.0 ms | 182.0 ms | 210.0 ms | 58.0 QPS | 0.00% |
| **250 VU** | 182.5 ms | 170.0 ms | 215.0 ms | 240.0 ms | 290.0 ms | 88.2 QPS | 0.02% |
| **500 VU** | 265.0 ms | 240.0 ms | 340.0 ms | 390.0 ms | 480.0 ms | 112.5 QPS | 0.12% |
