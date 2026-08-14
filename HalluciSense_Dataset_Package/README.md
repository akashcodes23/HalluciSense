# HalluciSense Canonical Benchmark Dataset Package

## Package Overview
This package contains the official, frozen multi-domain diagnostic benchmark dataset ($N=750$) and reproduction infrastructure for **HalluciSense: Confidence-Aware Hybrid Framework for Detecting and Quantifying Hallucinations in Large Language Models**.

### Package Contents
* `benchmark/benchmark_dataset.jsonl`: The canonical frozen $N=750$ evaluation benchmark in JSON Lines format.
* `DATASET_PROVENANCE.md`: Detailed attribution, source relationships, domain breakdown, and preprocessing history.
* `dataset_provenance_manifest.json`: Machine-readable dataset specification and metadata manifest.
* `dataset_hashes.json`: Cryptographic SHA-256 integrity checksums for all benchmark artifacts.
* `source_datasets/README.md`: Documentation of foundational literature and query pattern sources.
* `reproduction/reproduce_phase6_dataset.py`: Self-contained verification and reproduction script.

---

### Benchmark Quick Facts
* **Total Sample Count ($N$)**: 750 claims
* **Domains**: 15 distinct research disciplines (50 claims each)
* **Class Distribution**: 375 Factual (Label 0) / 375 Hallucinated (Label 1)
* **Canonical SHA-256**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`
* **License**: MIT License

---

### How to Verify & Reproduce
```bash
python reproduction/reproduce_phase6_dataset.py
```
Expected output confirms all 6 structural verification gates and exact cryptographic SHA-256 match.
