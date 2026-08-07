#!/usr/bin/env bash
# HalluciSense Fresh Machine Automated Installation & Benchmark Script (Part 2)
# Executes complete installation, testing, benchmarking, and paper generation without human intervention.

set -euo pipefail

echo "================================================================================"
echo "HALLUCISENSE FRESH MACHINE REPRODUCIBILITY INSTALLER"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "[1/6] Setting up isolated Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[2/6] Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt pytest structlog pandas scikit-learn scipy matplotlib psutil

export PYTHONPATH="${ROOT_DIR}/backend:${PYTHONPATH:-}"

echo "[3/6] Verifying HuggingFace model cache & dataset integrity..."
python3 -c "
from evaluation.datasets.public_benchmark_loaders import load_all_benchmark_datasets
datasets = load_all_benchmark_datasets(max_per_dataset=5)
print(f'Successfully loaded {len(datasets)} benchmark datasets!')
"

echo "[4/6] Running automated Pytest test suite..."
cd backend
pytest tests/test_phase26_benchmarks.py tests/test_production_api.py tests/test_three_pillars_complete.py tests/test_regression_v2.py -v

echo "[5/6] Executing master scientific benchmark runner..."
python3 evaluation/benchmark_runner.py

echo "[6/6] Verifying CI Quality Gates..."
python3 scripts/check_phase26_quality_gates.py

echo "================================================================================"
echo "✅ FRESH INSTALLATION & REPRODUCIBILITY SUITE COMPLETED SUCCESSFULLY!"
echo "================================================================================"
