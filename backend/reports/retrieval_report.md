# HalluciSense Retrieval Diagnostics Report (Phase 25)

## Executive Summary
Formal Information Retrieval (IR) evaluation executed across `1` benchmark queries using the hybrid retriever (Wikipedia REST API + BM25 + FAISS + CrossEncoder Reranker).

## IR Benchmark Metrics

| Metric | Empirical Score | Benchmark Target | Status |
|:---|:---:|:---:|:---:|
| **Recall@1** | `1.0000` | $\ge 0.70$ | ✅ |
| **Recall@3** | `1.0000` | $\ge 0.80$ | ✅ |
| **Recall@5** | `1.0000` | $\ge 0.85$ | ✅ |
| **Recall@10** | `1.0000` | $\ge 0.90$ | ✅ |
| **MRR** | `1.0000` | $\ge 0.75$ | ✅ |
| **nDCG@5** | `1.0000` | $\ge 0.80$ | ✅ |
| **MAP** | `0.5000` | $\ge 0.75$ | ⚠️ |
| **Evidence Coverage** | `1.0000` | $\ge 0.80$ | ✅ |
