# HalluciSense Artifact Execution Guide

## Execution Options

### 1. Fast Smoke Test (< 2 minutes)
Validates dataset loading, model inference, and metric calculations:
```bash
cd backend
pytest tests/test_phase26_benchmarks.py -v
```

### 2. Full Master Scientific Benchmark (< 15 minutes)
Executes evaluation across 11 public datasets, 10 baselines, 13 ablations, statistical significance tests, and generates all 600 DPI publication figures:
```bash
python3 backend/evaluation/benchmark_runner.py
```

### 3. CI Quality Gate Verification
```bash
python3 backend/scripts/check_phase26_quality_gates.py
```
