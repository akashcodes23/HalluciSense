# HalluciSense Dataset Provenance & Attribution Report

## 1. Provenance Statement
The **HalluciSense Canonical Benchmark Dataset ($N=750$)** is an independently curated multi-domain diagnostic evaluation suite developed for rigorous factuality and hallucination detection assessment.

### Provenance Classification:
* **Classification**: Project-Created Multi-Domain Diagnostic Suite with Formal Provenance Pipeline.
* **Primary Generation Engine**: `backend/evaluation/benchmark_dataset/importer.py` (`generate_publication_benchmark_dataset(n_per_domain=50, seed=42)`).
* **Reference Literature Foundations**:
  * **HaluEval** (Li et al., 2023): General knowledge, dialogue, and QA hallucination patterns.
  * **TruthfulQA** (Lin et al., 2022): Human misconception and adversarial falsehood patterns.
  * **FEVER** (Thorne et al., 2018): Claim-evidence entailment and contradiction verification principles.

---

## 2. Transformation Pipeline

```
[15 Canonical Domain Taxonomies]
               ↓
[Curated Factual & Contradicted Query-Response Pairs]
               ↓
[Deterministic Multi-Domain Expansion (50 samples / domain, Seed 42)]
               ↓
[Standardized BenchmarkExample Schema Normalization]
               ↓
[Deduplication Audit (0 Duplicate IDs, 0 Duplicate Claims)]
               ↓
[Cryptographic Integrity Freeze (SHA-256: dfe8c6e4...)]
```

---

## 3. Cryptographic Signatures

| File Path | Records | Byte Size | SHA-256 Hash |
|---|---|---|---|
| `benchmark/benchmark_dataset.jsonl` | 750 | 295,354 | `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5` |
| `dataset_provenance_manifest.json` | 1 | 2,841 | `9b128ef87a5611f7c1703f269a84a6c0b3967aa52b801579d4692797e8b6159c` |

---

## 4. Ground-Truth Labeling & Verification
* **Label 0 (Factual)**: Claim aligns completely with verifiable empirical facts, scientific consensus, or historical records.
* **Label 1 (Hallucinated)**: Claim introduces explicit factual contradictions, temporal impossibilities, or fabricated relationships.
* **Class Balance**: Exactly 375 Factual (50.0%) and 375 Hallucinated (50.0%).
