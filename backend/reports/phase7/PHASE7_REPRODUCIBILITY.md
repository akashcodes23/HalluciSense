# HalluciSense Phase 7 — Experiment Reproducibility Guide

## 1. Reproduction Command & Environment

```bash
# 1. Ensure local Ollama instance is active with qwen2.5-coder:1.5b
ollama list

# 2. Re-compile Phase 7 artifacts from persisted 750 traces:
PYTHONPATH=backend ./venv/bin/python backend/evaluation/generate_phase7_artifacts_from_traces.py

# 3. Or re-run the live benchmark pipeline (estimated 2.5 hours for N=750 live multi-generation):
PYTHONPATH=backend ./venv/bin/python backend/evaluation/run_phase7_live_three_pillar_benchmark.py --provider ollama --model qwen2.5-coder:1.5b --p3-count 3
```

## 2. Environment Metadata
* **Python**: 3.10.12
* **PyTorch Device**: `mps` (Apple Silicon GPU Acceleration)
* **Local Provider**: Ollama (REST API `http://127.0.0.1:11434`)
* **Primary LLM**: `qwen2.5-coder:1.5b` (Temperature: 0.70)
* **P1 Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
* **P1 / P3 Cross-Encoder**: `cross-encoder/nli-deberta-v3-small`
* **Canonical Prompts**: `backend/evaluation/results/benchmark_dataset.jsonl` ($N=750$)
* **Canonical SHA-256**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`
