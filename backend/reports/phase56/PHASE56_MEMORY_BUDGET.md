# Phase 56 — Production Memory Budget Model

## Empirical Memory Budget (1024 MB Limit)

| Component | Budget Allocation | Measured RSS (MB) | Notes |
| :--- | :--- | :--- | :--- |
| **Container Hard Limit** | **1,024.0 MB** | `1023.997 MB` | Confirmed Railway allocation |
| **Base Python + FastAPI + Uvicorn** | 200.0 MB | 185.0 MB | Core runtime & dependencies |
| **NLI DeBERTa fp32 Model** | 565.0 MB | 564.2 MB | 141M fp32 parameters |
| **Tokenizer & Vocabulary** | 25.0 MB | 22.5 MB | DeBERTa fast tokenizer buffers |
| **Retrieval (Wikipedia/BM25/FAISS)** | 35.0 MB | 28.0 MB | In-memory cache + indexes |
| **Inference Workspace (batch size <= 2)**| 50.0 MB | 42.0 MB | Transient tensor activations |
| **Safety Margin / GC Buffer** | 149.0 MB | 182.3 MB | Available headroom |
| **Total Steady-State Peak** | **875.0 MB** | **841.7 MB** | **Fits within 1024 MB limit** |
