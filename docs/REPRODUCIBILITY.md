# HalluciSense Reproducibility Protocol

## 1. Frozen Benchmark Artifacts
- **Canonical Dataset:** `backend/evaluation/results/benchmark_dataset.jsonl`
- **Canonical SHA-256 Checksum:** `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`
- **Dataset Cardinality:** $N = 750$ verified multi-domain evaluation claims.

## 2. Environment & Dependency Freeze
- **Python Version:** 3.10.x
- **Random Seed:** 42 (enforced across NumPy, PyTorch, and random samplers)
- **Model Registry Singletons:**
  * NLI Model: `cross-encoder/nli-deberta-v3-small`
  * Embeddings: `all-MiniLM-L6-v2`
  * Re-ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## 3. One-Command Evaluation Reproduction
```bash
PYTHONPATH=backend backend/venv/bin/python backend/evaluation/run_comprehensive_research_evaluation.py
```
This generates:
- `backend/reports/research_ablation_matrix.json`
- `backend/reports/research_baseline_comparison.json`
- `backend/reports/research_calibration_report.json`
- `backend/reports/research_closed_loop_metrics.json`
- `experiment_manifest.json`
