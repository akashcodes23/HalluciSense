# PHASE 49 — COMPLETE ALLOCATION GRAPH
**Module-by-Module Memory Attribution & Dependency Footprint**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `AUDITED & ISOLATED`

---

## 1. Static Import Memory Footprint

```
+--------------------------------------------------------------------------------------------------------+
| Component / Library            | Cold RSS  | Warm RSS  | Request Temp | Persistent | Can Be Removed/Quant?|
+--------------------------------------------------------------------------------------------------------+
| Python Runtime + Standard Lib  |  15.78 MB |  15.78 MB | < 1 MB       | Yes        | No                   |
| numpy / scipy                  |  35.14 MB |  35.14 MB | 2 MB         | Yes        | Core Dependency      |
| scikit-learn (HistGradBoost)   | 117.38 MB | 125.00 MB | 1 MB         | Yes (Frozen| Invariant (Frozen)   |
| PyTorch Core (CPU)             | 243.12 MB | 243.12 MB | 15 MB        | Yes        | Single Thread CPU    |
| Transformers (Tokenizer/DeBERTa| 335.20 MB | 524.65 MB | 25 MB        | Yes        | Singleton Bound      |
| FastAPI / Starlette / Uvicorn  | 368.91 MB | 368.91 MB | 3 MB         | Yes        | Web Framework        |
| SentenceTransformer (Removed)  |   0.00 MB |   0.00 MB | 0 MB         | NO (Removed| REMOVED              |
| CrossEncoder Reranker (Disabled|   0.00 MB |   0.00 MB | 0 MB         | NO (Disable| DISABLED             |
+--------------------------------------------------------------------------------------------------------+
```

---

## 2. Transient Memory Spike Attribution (~295 MB)

The investigation isolated the exact origin of the 295 MB transient memory delta:
1. **DeBERTa Relative Attention Matrices**: During intra-response claim pairing without chunking, a 15-claim text spawned 105 NLI pairs in a single forward pass. DeBERTa-v3 relative position matrices $(2 \times \text{seq\_len}) \times \text{seq\_len}$ across 105 pairs consumed **~210 MB** of heap scratchpad.
2. **PyTorch Caching Allocator Workspaces**: PyTorch CPU allocator held intermediate layer norm and linear workspace buffers across batch loops (**~60 MB**).
3. **Trace Object Serialization**: Raw tensors and long un-truncated evidence strings retained in diagnostic structures (**~25 MB**).

### Remediation:
- Micro-chunking (batch size $\le 2$).
- Claim limit in P3 ($\le 8$ claims, max 28 pairs).
- Immediate `del inputs, logits` under `torch.inference_mode()`.
- Recursive trace sanitization removing any tensor or large ndarray.
