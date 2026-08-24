# Phase 13 — Repository Forensic Audit Report

## 1. Executive Summary
This forensic audit was conducted to inspect data flow, training/evaluation boundaries, and model parameter selection across the HalluciSense repository before publication hardening.

## 2. Forensic Questions & Grounded Findings

| Question | Forensic Finding | Evidence / Code Path | Risk Level |
| :--- | :--- | :--- | :--- |
| **1. Training Data** | Zero internal model training is performed on the benchmark dataset. HalluciSense utilizes frozen pre-trained weights (`cross-encoder/nli-deberta-v3-small`, `all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`). | [`backend/app/core/engine/model_registry.py`](file:///Users/akashgpatil/major_project/backend/app/core/engine/model_registry.py) | **NONE** |
| **2. Calibration Data** | Platt parameters ($a=1.82, b=-0.45$) and Isotonic regression mappings were fitted on the 70% development partition. | [`backend/app/core/engine/calibration.py`](file:///Users/akashgpatil/major_project/backend/app/core/engine/calibration.py) | **LOW** |
| **3. Threshold Selection** | Decision thresholds ($H < 0.35$ for verified, $H \ge 0.50$ for high risk) are fixed operational bounds set during system calibration. | [`backend/app/core/engine/fusion.py`](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py) | **LOW** |
| **4. Fusion Parameter Selection** | Canonical weights ($\alpha=0.40, \beta=0.30, \gamma=0.30$) are fixed theoretical defaults. Adaptive weights re-normalize dynamically per query without test label access. | [`backend/app/core/engine/fusion.py`](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py) | **NONE** |
| **5. Abstention Thresholds** | Selective abstention triggers on $S_{\text{evidence}} < 0.40$ or ambiguity $|H - 0.40| < 0.08$ with epistemic uncertainty $> 0.75$. | [`backend/app/core/engine/calibration.py`](file:///Users/akashgpatil/major_project/backend/app/core/engine/calibration.py) | **NONE** |
| **6. Evaluation Data** | Canonical benchmark $N=750$ claims in `benchmark_dataset.jsonl` (SHA-256: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`). | [`backend/evaluation/results/benchmark_dataset.jsonl`](file:///Users/akashgpatil/major_project/backend/evaluation/results/benchmark_dataset.jsonl) | **LOW** |
| **7. Retrieval Data** | Hybrid retrieval queries Wikipedia via REST and internal FAISS index of scientific reference texts. Does not access test labels. | [`backend/app/modules/knowledge/retriever.py`](file:///Users/akashgpatil/major_project/backend/app/modules/knowledge/retriever.py) | **NONE** |
| **8. Correction Data** | Correction engine uses retrieved factual evidence and deterministic repair policies (units, negations, causal inversion). | [`backend/app/core/correction/correction_engine.py`](file:///Users/akashgpatil/major_project/backend/app/core/correction/correction_engine.py) | **NONE** |
| **9. Label Leakage** | No ground truth labels, categories, or expected verdicts enter the pipeline inference path. | Full pipeline audit across `backend/app/core/engine/` | **NONE** |
| **10. Benchmark-Specific Rules** | Symbolic checkers (units, negations, causal direction) operate on linguistic and physical invariants, not benchmark dataset IDs or synthetic keys. | [`backend/app/core/engine/numeric_unit_checker.py`](file:///Users/akashgpatil/major_project/backend/app/core/engine/numeric_unit_checker.py) | **LOW** |
