# Phase 56 — Root Cause Classification

## Primary Classification: **R1 — Railway Memory Limit Exceeded (Triggered by R2: PyTorch/Transformers Deserialization Transient Memory Spike)**

### The Mechanism
1. **WHAT Happened**: The backend container was terminated by the Linux kernel with `SIGKILL (Exit Code 137)`.
2. **WHEN It Happened**: During background warmup execution of `AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")`.
3. **WHY It Happened**: Standard HuggingFace model loading deserialized the full unquantized checkpoint into an in-memory dictionary while simultaneously allocating module weight tensors. This doubled transient RAM requirements to **1,107 MB – 1,672 MB**, exceeding Railway's **1,023.99 MB** container limit.
4. **PROVING EVIDENCE**:
   - Railway API metrics confirmed limit = `1023.997 MB`, max peak = `1672.82 MB`.
   - Time-series showed allocation rising from 567 MB -> 893 MB -> 1,107 MB at `06:42 UTC` followed immediately by restart.
   - Absence of Python exception tracebacks confirms external SIGKILL termination.
