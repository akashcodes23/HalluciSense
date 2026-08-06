# HalluciSense Artifact Evaluation & Reproducibility Guide

This document provides complete instructions for verifying and reproducing all empirical results, figures, statistical tests, and LaTeX paper tables presented in our manuscript for top-tier AI conference and journal artifact evaluations (ACL, EMNLP, NeurIPS, IEEE, Elsevier).

---

## 🚀 Quickstart: Single-Command Reproduction

To regenerate the entire scientific benchmark, probability calibration, statistical validation suite, 300 DPI figures, and LaTeX manuscript from scratch using a single command:

```bash
chmod +x ./reproduce.sh
./reproduce.sh
```

---

## ⚙️ System Requirements & Hardware SLA

- **Operating System**: macOS (Apple Silicon / Intel), Linux (Ubuntu 20.04/22.04), or Windows (WSL2).
- **RAM Footprint**: Minimum 8 GB system RAM (Inference RSS RAM footprint $< 512$ MB SLA).
- **CPU / GPU**: Executable on standard multi-core CPUs. Supports Apple Silicon MPS, PyTorch CUDA 12.1, or CPU fallback execution.
- **Expected Execution Runtime**: **~28.96 seconds** end-to-end ($N=750$ claims across 15 research domains).
- **Fixed Random Seed**: $S = 42$ (Deterministic RNG seeding across Python, NumPy, PyTorch, and CUDA).

---

## 📦 Dependency Manifests & Environment Setup

HalluciSense locks every dependency across 4 package management formats:

### 1. Conda Environment Setup
```bash
conda env create -f environment.yml
conda activate hallucisense-research
```

### 2. Pip Locked Requirements
```bash
cd backend
pip install -r requirements-lock.txt
```

### 3. Poetry Environment Setup
```bash
poetry install
poetry shell
```

### 4. Docker Container Build
```bash
docker compose build
docker compose up -d
```

---

## 📊 Dataset Verification & SHA256 Checksums

Public benchmark dataset checksums are verified automatically during execution against `backend/evaluation/results/dataset_checksums.json`:

| Dataset | Research Domain | Sample Count ($N$) | License |
| :--- | :--- | :---: | :---: |
| **TruthfulQA** | Misconceptions & Miscalibration | 100 | Apache 2.0 |
| **FEVER** | Fact Extraction & Verification | 120 | CC BY-SA 4.0 |
| **SciFact** | Scientific Claim Verification | 100 | CC BY-NC 4.0 |
| **FreshQA** | Fast-Changing Temporal Knowledge | 80 | MIT |
| **FactScore** | Long-Form Atomic Precision | 100 | MIT |
| **RAGTruth** | RAG Hallucination Detection | 100 | Apache 2.0 |
| **HaluEval** | General QA & Dialogue | 150 | MIT |

---

## 📂 Output Deliverable Artifacts

After running `./reproduce.sh`, generated deliverables are available at:

- **JSON Predictions**: `backend/evaluation/results/predictions.json`
- **CSV Predictions**: `backend/evaluation/results/predictions.csv`
- **Parquet Predictions**: `backend/evaluation/results/predictions.parquet`
- **Master Evaluation Report**: [reports/evaluation_report.md](file:///Users/akashgpatil/major_project/backend/reports/evaluation_report.md)
- **Statistical Significance Report**: [reports/statistics_report.md](file:///Users/akashgpatil/major_project/backend/reports/statistics_report.md)
- **Elsevier LaTeX Manuscript**: `backend/paper/elsevier_manuscript.tex`
- **LaTeX Publication Tables**: `backend/paper/ablation_tables.tex` & `backend/paper/publication_tables.tex`
- **300 DPI Publication Figures**: `backend/evaluation/figures/` & `backend/evaluation/calibration_figures/`
