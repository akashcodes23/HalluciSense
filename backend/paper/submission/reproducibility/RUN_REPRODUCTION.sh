#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "HALLUCISENSE AUTOMATED REPRODUCIBILITY EXECUTION SCRIPT"
echo "================================================================================"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH=backend

echo "Step 1: Verifying Canonical Benchmark Dataset SHA-256 Hash..."
EXPECTED_SHA="dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"
OBSERVED_SHA=$(shasum -a 256 backend/evaluation/results/benchmark_dataset.jsonl | awk '{print $1}')

if [ "$OBSERVED_SHA" != "$EXPECTED_SHA" ]; then
    echo "[FATAL] Benchmark SHA-256 mismatch! Expected: $EXPECTED_SHA, Observed: $OBSERVED_SHA"
    exit 1
fi
echo "[PASS] Canonical Benchmark SHA-256 Hash Verified: $OBSERVED_SHA"

echo ""
echo "Step 2: Executing Clean-Room Reproduction Audit..."
backend/venv/bin/python backend/evaluation/phase15/run_reproducibility_check.py

echo ""
echo "Step 3: Running Final Claim & Metric Consistency Audit..."
backend/venv/bin/python backend/evaluation/phase19/final_claim_consistency.py

echo ""
echo "Step 4: Executing Full Regression Test Suite (60+ Tests)..."
backend/venv/bin/pytest backend/tests/ -v

echo ""
echo "================================================================================"
echo "ALL REPRODUCIBILITY CHECKS AND REGRESSION TESTS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
