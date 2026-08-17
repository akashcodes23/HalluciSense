# Phase 8 Reproducibility Guide

## Reproduction Steps

```bash
# 1. Generate Dataset 8A and Verify Manifest Hash
PYTHONPATH=backend python3 backend/evaluation/phase8a/build_dataset_8a.py

# 2. Run Phase 8A Baseline Evaluation
PYTHONPATH=backend python3 backend/evaluation/phase8a/run_phase8a_evaluation.py

# 3. Reorganise Phase 8B Audit Artifacts
PYTHONPATH=backend python3 backend/evaluation/phase8b/build_report_8b.py

# 4. Run Phase 8C Controlled Stress Test
PYTHONPATH=backend python3 backend/evaluation/phase8c/run_phase8c_evaluation.py

# 5. Run Enhanced P1 Evaluation
PYTHONPATH=backend python3 backend/evaluation/phase8a/run_phase8a_enhanced.py

# 6. Generate Master Publication Figures and Artifacts
PYTHONPATH=backend python3 backend/evaluation/phase8/generate_phase8_publication_artifacts.py

# 7. Execute Test Suite
PYTHONPATH=backend pytest backend/tests/test_phase8a_adversarial.py backend/tests/test_phase8_enhanced_p1.py -v
```
