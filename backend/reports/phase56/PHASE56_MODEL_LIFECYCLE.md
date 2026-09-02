# Phase 56 — Model Lifecycle Forensics

## Runtime Object Inventory

| Component | Model / Class | Instance Count | Storage Mode | Lifecycle |
| :--- | :--- | :--- | :--- | :--- |
| **NLI CrossEncoder** | `cross-encoder/nli-deberta-v3-small` | **1** | Process Memory (Singleton) | Lazy / Warmup initialized |
| **Tokenizer** | `DebertaV2TokenizerFast` | **1** | Process Memory (Singleton) | Shared with NLI model |
| **SentenceTransformer** | `all-MiniLM-L6-v2` | **0** | Not loaded in production path | Eliminated in Phase 48 |
| **Vector Store** | `FAISSVectorStore` | **1** | In-memory lightweight mock list | Singleton per pipeline |
| **BM25 Index** | `BM25Retriever` | **1** | In-memory tokenized dict | Singleton per pipeline |
| **Reranker** | `CrossEncoderReranker` | **0** | Not loaded (disabled by default) | On-demand if enabled |
| **Workers** | `uvicorn` | **1** | Single OS process | `workers=1` in `start.py` |
| **PyTorch Threads** | `torch.set_num_threads` | **1** | Single CPU thread | Bounded to 1 |

### Key Lifecycle Findings
1. Singleton enforcement via `threading.RLock()` in `ModelRegistry` successfully guarantees that exactly **one** instance of `cross-encoder/nli-deberta-v3-small` is created per process.
2. There are no multiple process forks or worker duplication.
