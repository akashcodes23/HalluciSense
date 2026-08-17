# HalluciSense Phase 7B — Forensic Reproducibility Guide

## 1. Reproduction Command
To reproduce the complete Phase 7B forensic analysis from existing Phase 6 and Phase 7 artifacts:

```bash
PYTHONPATH=backend ./venv/bin/python backend/evaluation/run_phase7b_integrity_analysis.py
```

## 2. Cryptographic Manifest
* **Benchmark Dataset SHA-256**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`
* **Phase 6 Canonical Traces**: 750 JSON files in `backend/reports/phase6/traces/`
* **Phase 7 Live Traces**: 750 JSON files in `backend/reports/phase7/traces/`
* **Train / Val Random Seed**: `42`
* **Bootstrap Random Seed**: `42`
* **Python Runtime**: `3.10.12`
