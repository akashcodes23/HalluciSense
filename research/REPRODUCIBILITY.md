# HalluciSense Scientific Reproducibility Package

**Framework**: HalluciSense: An Availability-Aware, Calibrated Multi-Signal Verification Framework  
**Verification Level**: 100% Deterministic & Clean-Room Reproducible  
**Target Environment**: macOS / Linux (Ubuntu 22.04+), Python 3.10.x  

---

## 1. System Dependencies & Environment Setup

### 1.1 Python Environment
```bash
# 1. Create and activate dedicated virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 2. Upgrade core packaging tools
pip install --upgrade pip setuptools wheel

# 3. Install locked dependencies
pip install -r requirements-lock.txt
# Alternatively:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.47.1 sentence-transformers==3.3.1 faiss-cpu==1.9.0 rank-bm25==0.2.2
pip install scikit-learn==1.7.2 scipy==1.15.3 pandas==2.3.3 numpy==1.26.4
pip install fastapi==0.115.6 uvicorn==0.32.1 pydantic==2.10.4 pypdf==6.21.2 pytest==8.3.4
```

### 1.2 LaTeX Compilation Engine
To build the IEEE conference PDF manuscript without installing full multi-gigabyte TeX distributions:
```bash
# macOS (via Homebrew)
brew install tectonic

# Linux (via Cargo / Binary release)
curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.tectonic-typesetting.github.io/install.sh | sh
```

---

## 2. Dataset Verification & Provenance

The canonical diagnostic benchmark ($N=750$) is cryptographically locked with SHA-256:
```bash
# Verify integrity of canonical benchmark dataset
shasum -a 256 backend/evaluation/results/benchmark_dataset.jsonl
# Expected output:
# dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5
```

---

## 3. Reproduction Commands

### 3.1 Running the Manuscript Integrity & Regression Tests
```bash
venv/bin/pytest backend/tests/test_phase17_manuscript_integrity.py \
               backend/tests/test_phase16_scientific_gate.py \
               backend/tests/test_phase15_scientific_package.py \
               backend/tests/test_phase14_signal_availability.py \
               backend/tests/test_phase14_calibration.py \
               backend/tests/test_phase14_abstention.py \
               backend/tests/test_phase14_correction.py
```

### 3.2 Running the Full Scientific Benchmark & Statistical Analysis
```bash
# Execute statistical testing and bootstrap confidence interval generation
python backend/evaluation/phase15/statistical_analysis.py

# Execute availability robustness evaluation across 7 signal masks
python backend/evaluation/phase15/adaptive_fusion_statistics.py

# Execute baseline comparison scripts
python backend/evaluation/phase15/baseline_comparison.py
```

### 3.3 Regenerating Manuscript LaTeX Tables
```bash
python backend/evaluation/phase17/generate_latex_tables.py
```

### 3.4 Compiling the IEEE Conference Paper PDF
```bash
# Build the final publication PDF from paper/main.tex
cd paper && tectonic main.tex -o . && mv main.pdf HalluciSense_IEEE_Research_Paper.pdf
```

---

## 4. Reproducibility Configuration & Random Seeds

- **Random Seed**: Fixed to `42` across all data splitters, bootstrap resamplers, and stochastic generators (`app/core/engine/seed.py`).
- **Bootstrap Iterations**: $B = 10,000$ for internal confidence intervals; $B = 500$ for external benchmark paired Wilcoxon/bootstraps.
- **Operating Decision Threshold**: $\tau^* = 0.54$ (fitted on development partition).
- **Platt Calibration Parameters**: $a = 1.82$, $b = -0.45$ (fitted on internal validation split).
- **Selective Abstention Thresholds**: $S_{\text{evidence}} < 0.40$ or $|H - 0.40| < 0.08$.
- **Reverification Acceptance Gate**: $H_{\text{reverify}} < 0.20$.
- **Peak Memory Guarantee**: $\le 1.2\text{ GB}$ RAM via singleton caching in `ModelRegistry`.
