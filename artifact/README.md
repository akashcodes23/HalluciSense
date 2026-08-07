# HalluciSense Artifact Evaluation Guide

Welcome to the HalluciSense Artifact Evaluation package. This repository provides a complete, self-contained, reproducible benchmarking framework designed for peer review by committee members at NeurIPS, ICLR, ICML, ACL, EMNLP, and Elsevier top-tier Q1 journals.

---

## Artifact Contents

- **`INSTALL.md`**: Step-by-step installation instructions for Linux, macOS, and Docker.
- **`RUN.md`**: Commands for running fast smoke tests (< 2 minutes) and full reproduction (< 30 minutes).
- **`CHECKSUMS.md`**: SHA256 checksums verifying file integrity across datasets, models, and evaluation outputs.
- **`LICENSE`**: MIT Open Science license.
- **`expected_outputs/`**: Reference outputs generated during primary benchmark execution.

---

## Quick Start (One-Line Execution)

```bash
# Clean machine automated reproduction
bash scripts/fresh_install.sh
```

Or via Docker:
```bash
docker compose up --build
```
