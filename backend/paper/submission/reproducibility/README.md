# HalluciSense Reproducibility Package

This directory contains the master reproducibility configuration, manifests, and automated execution scripts for replicating the empirical results presented in the manuscript.

## Quick Reproduction Command
To execute the automated clean-room verification and test suite:
```bash
bash backend/paper/submission/reproducibility/RUN_REPRODUCTION.sh
```

## Directory Inventory
- `RUN_REPRODUCTION.sh`: Shell script to execute full regression and clean-room audits.
- `requirements.txt`: Exact pip dependency freeze.
- `environment.yml`: Conda environment definition.
- `REPRODUCIBILITY_MANIFEST.json`: Master execution metadata and environment configuration.
- `MODEL_MANIFEST.json`: Model weights, registry keys, and initialization parameters.
- `DATASET_MANIFEST.json`: Benchmark hashes, splits, and external dataset licenses.
