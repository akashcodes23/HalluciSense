# Phase 56 — Environment Parity

## Local vs Railway Comparison

| Attribute | Local Development | Railway Production | Parity Impact |
| :--- | :--- | :--- | :--- |
| **OS** | macOS Darwin 24.6.0 (ARM64) | Linux (x86_64 Debian/Ubuntu container) | Low |
| **Python** | Python 3.10.12 | Python 3.11 (Slim Docker image) | Low |
| **PyTorch** | 2.6.0 (CPU) | 2.6.0+cpu (Linux wheel) | Identical |
| **Transformers** | 4.49.0 | 4.49.0 | Identical |
| **Container Limit** | Host RAM (16 GB+) | **1024 MB (Hard Limit)** | **CRITICAL** |
| **Allocators** | macOS jemalloc | Linux glibc ptmalloc | Requires `malloc_trim` |
| **Process Model** | 1 worker, 1 thread | 1 worker, 1 thread | Identical |
| **NLI Model** | `cross-encoder/nli-deberta-v3-small` | `cross-encoder/nli-deberta-v3-small` | Identical |
