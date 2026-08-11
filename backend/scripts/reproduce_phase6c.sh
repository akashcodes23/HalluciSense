#!/usr/bin/env bash
# Phase 6C Reproducibility Script
# Usage: cd backend && bash scripts/reproduce_phase6c.sh
# Reproduces all Phase 6C experiments from scratch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BACKEND_DIR"

echo "========================================================"
echo "PHASE 6C REPRODUCIBILITY VERIFICATION"
echo "========================================================"
echo "Repository: $(git remote get-url origin 2>/dev/null || echo 'local')"
echo "Git SHA: $(git rev-parse HEAD)"
echo "Git Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Python: $(python3 --version)"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# 1. Verify frozen SHA
EXPECTED_SHA="cbe4de7"
CURRENT_SHA="$(git rev-parse --short HEAD)"
if [ "$CURRENT_SHA" != "$EXPECTED_SHA" ]; then
    echo "WARNING: Current SHA $CURRENT_SHA differs from frozen SHA $EXPECTED_SHA"
    echo "Results may differ from Phase 6C reported values."
else
    echo "SHA VERIFIED: $CURRENT_SHA == $EXPECTED_SHA"
fi
echo ""

# 2. Run metric consistency tests (MUST pass before anything else)
echo "--- STEP 1: Metric Consistency Gate ---"
python3 -m pytest tests/test_metrics_consistency.py -v --tb=short -q
echo "Metric consistency: PASSED"
echo ""

# 3. Run Phase 6C publication evaluation
echo "--- STEP 2: Phase 6C Publication Evaluation ---"
python3 -m scripts.run_phase6c_publication_eval --bootstrap 5000 --robustness-n 30
echo ""

# 4. Run full regression suite
echo "--- STEP 3: Full Regression Suite ---"
python3 -m pytest tests/ -x --timeout=120 -q --ignore=tests/integration
echo ""

# 5. Verify all report files were generated
echo "--- STEP 4: Report File Verification ---"
REQUIRED_FILES=(
    "reports/phase6c/experiment_manifest.json"
    "reports/phase6c/ablation_results.json"
    "reports/phase6c/domain_results.json"
    "reports/phase6c/robustness_results.json"
    "reports/phase6c/statistical_results.json"
    "reports/phase6c/error_transitions.json"
    "reports/phase6c/temporal_adversarial_results.json"
    "reports/phase6c/modality_results.json"
    "reports/phase6c/phase6c_ablation_results.md"
    "reports/phase6c/phase6c_domain_results.md"
    "reports/phase6c/phase6c_robustness_results.md"
    "reports/phase6c/phase6c_temporal_adversarial_results.md"
    "reports/phase6c/phase6c_statistical_results.md"
    "reports/phase6c/phase6c_error_analysis.md"
    "reports/phase6c/phase6c_reproducibility_report.md"
    "reports/phase6c/phase6c_architecture_freeze.md"
    "reports/phase6c/phase6c_dataset_inventory.json"
    "reports/phase6c/phase6c_phase6b_metric_audit.md"
)
ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        ALL_OK=false
    fi
done

if $ALL_OK; then
    echo ""
    echo "========================================================"
    echo "PHASE 6C REPRODUCTION: COMPLETE"
    echo "All reports generated. Results are in reports/phase6c/"
    echo "========================================================"
else
    echo ""
    echo "========================================================"
    echo "PHASE 6C REPRODUCTION: INCOMPLETE - Missing report files"
    echo "========================================================"
    exit 1
fi
