# Phase 56 — Concurrency Forensics

## Concurrency Architecture

- **Uvicorn Worker Count**: `1` (Single process, asynchronous event loop).
- **Inference Semaphore**: `ModelRegistry.get_nli_semaphore(max_concurrent=1)`.
- **Batch Size**: Strictly bounded to `batch_size=2` in `EvidenceEntailmentEngine.classify_batch()`.
- **Sequence Length**: Truncated at `256` tokens for evidence and `128` tokens for claims.

### Concurrency Memory Scaling
Under concurrent request load:
$$\text{Peak Memory} \approx \text{Baseline RSS} + 1 \times \text{Inference Workspace} \approx 620\text{ MB} + 45\text{ MB} = 665\text{ MB}$$
Because `max_concurrent=1` is enforced via semaphore, additional requests queue rather than multiplying memory workspace allocations.
