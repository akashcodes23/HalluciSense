# HalluciSense: A Confidence-Aware Hybrid Framework for Detecting and Quantifying Hallucinations in LLMs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com)
[![Status](https://img.shields.io/badge/Status-FROZEN_%26_VALIDATED-success.svg)](#)

HalluciSense is a production-quality, hybrid hallucination detection framework that unifies **External Evidence Grounding** (Pillar 1) and **Internal Structural Consistency** (Pillar 2) into a confidence-aware meta-classifier.

---

## Key Highlights

- **Pillar 1 (Evidence Consistency)**: Natural Language Inference (NLI) claim grounding against external reference documents ($\text{ROC-AUC} \approx 0.6260$).
- **Pillar 2 (Structural Consistency)**: Bidirectional pairwise NLI inter-claim contradiction matrix ($\text{ROC-AUC} \approx 0.6370$ on DEV).
- **Hybrid Fusion (Candidate 5)**: Gradient boosted meta-classifier combining 19 evidence, structural, and probability signals ($\text{ROC-AUC} = 0.6558$ on held-out validation, $p < 10^{-15}$ superior over Pillar 1).

---

## Quickstart

```bash
# 1. Clone & Activate Environment
git clone https://github.com/akashgpatil/hallucisense.git
cd hallucisense/backend
source venv/bin/activate

# 2. Run Test Suite (30/30 Pass)
python -m pytest tests/test_phase7_packaging.py -v

# 3. Launch REST API Server
python -m uvicorn app.main:app --reload --port 8000
```

Access API Documentation at: `http://localhost:8000/docs`

---

## Documentation Index

- 📖 [Installation Guide](docs/INSTALL.md)
- 🔬 [100% Reproducibility Steps](docs/REPRODUCIBILITY.md)
- 📊 [Dataset Dictionary & Fingerprints](docs/DATASET.md)
- 🏗️ [System Architecture](docs/ARCHITECTURE.md)
- 🏷️ [Model Card & Evaluation Metrics](docs/MODEL_CARD.md)
- ⚠️ [Limitations & Distribution Shift Notes](docs/LIMITATIONS.md)
- 🛡️ [AI Ethics & Guidelines](docs/ETHICS.md)
- 📜 [Changelog](docs/CHANGELOG.md)

---

## Citation

If you find HalluciSense useful in your research, please cite:

```bibtex
@software{Patil_HalluciSense_2026,
  author = {Patil, Akash G.},
  title = {HalluciSense: A Confidence-Aware Hybrid Framework for Detecting and Quantifying Hallucinations in LLMs},
  year = {2026},
  url = {https://github.com/akashgpatil/hallucisense}
}
```
