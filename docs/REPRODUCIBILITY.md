# HalluciSense Reproducibility Guide

All experiments in HalluciSense are 100% deterministic and reproducible.

## Deterministic Verification

1. **Random Seed**: Fixed at `RANDOM_STATE = 42` across all preprocessing, splitting, and classifier initialization.
2. **Phase 6M.1 Preflight**: Run `python -m evaluation.phase6m.run_phase6m_1` to verify feature matrix integrity.
3. **Phase 6M.2 Model Selection**: Run `python -m evaluation.phase6m.run_phase6m_2` to reproduce 5-fold 3-repeat cross-validation ($N=58,002$).
4. **Phase 6M.3 Held-Out Validation**: Run `python -m evaluation.phase6m.run_phase6m_3` to evaluate single-pass held-out validation ($N=12,483$).
5. **Phase 6M.4 Forensic Investigation**: Run `python -m evaluation.phase6m.run_phase6m_4` to reproduce all 9 diagnostic forensic stages.
