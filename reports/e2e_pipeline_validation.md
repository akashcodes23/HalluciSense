# Phase 25 Stage 2 — End-to-End Pipeline Latency Profile Report

## Stage-by-Stage Latency Breakdown

| Pipeline Stage | Measured Latency | SLA Target | Percentage |
| :--- | :---: | :---: | :---: |
| **Prompt Parsing** | 2.1 ms | &lt; 50 ms | 1.5% |
| **Claim Extraction** | 28.5 ms | &lt; 50 ms | 20.3% |
| **Evidence Retrieval** | 45.0 ms | &lt; 50 ms | 32.0% |
| **Crossencoder Reranking** | 32.4 ms | &lt; 50 ms | 23.1% |
| **Pillar1 Grounding** | 12.2 ms | &lt; 50 ms | 8.7% |
| **Pillar2 Self Consistency** | 14.8 ms | &lt; 50 ms | 10.5% |
| **Hybrid Fusion** | 1.2 ms | &lt; 50 ms | 0.9% |
| **Calibration Thresholding** | 0.5 ms | &lt; 50 ms | 0.4% |
| **Explanation Engine** | 3.4 ms | &lt; 50 ms | 2.4% |
| **Api Serialization** | 0.4 ms | &lt; 50 ms | 0.3% |

**Total End-to-End Inference Latency**: **140.5 ms** (&lt; 200 ms SLA Target).
