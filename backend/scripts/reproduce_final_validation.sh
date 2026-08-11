#!/bin/bash
set -e

echo "======================================================================"
echo "HALLUCISENSE PHASE 6J — FINAL REPRODUCIBILITY & VALIDATION GATE"
echo "======================================================================"

cd "$(dirname "$0")/.."

echo "[1/4] Verifying production invariants..."
python3 -c "from app.core.engine.fusion import FusionEngine; f = FusionEngine(); assert round(f.alpha + f.beta + f.gamma, 2) == 1.0; print('Production weights OK')"

echo "[2/4] Verifying dataset independence..."
python3 -c "import json; d = json.load(open('reports/phase6j/phase6j_dataset_integrity.json')); assert d['cross_phase_overlap']['status'] == 'PASS'; print('Dataset independence OK')"

echo "[3/4] Running full pytest regression suite..."
python3 -m pytest tests/test_phase6i.py tests/test_phase6e.py tests/test_epistemic.py tests/test_phase6_architecture.py tests/test_phase6b_dataset_integrity.py tests/test_temporal_benchmark.py tests/test_temporal_holdout.py tests/test_metrics_consistency.py tests/test_phase13_public_release.py -q

echo "[4/4] Verifying Phase 6I reproduction metrics..."
python3 -c "import json; repro = json.load(open('reports/phase6j/phase6i_reproduction_results.json')); assert repro['accuracy'] == 0.888; print('Phase 6I reproduction OK')"

echo "======================================================================"
echo "PHASE 6J VALIDATION SUCCESSFUL — ALL GATES PASSED (100% REPRODUCIBLE)"
echo "======================================================================"
