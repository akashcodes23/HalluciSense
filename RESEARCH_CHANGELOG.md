# HalluciSense Research & Engineering Changelog

## Version 2.0 (Research & Production Architecture Synchronization)

### 1. Architectural Standardization
- Preserved the three-pillar research formulation ($H = \alpha \cdot \text{FE} + \beta \cdot \text{CG} + \gamma \cdot \text{CF}$).
- Formalized Mode A (Canonical Fixed Baseline) and Mode B (Availability-Aware Adaptive Fusion with empirical reliability weights).
- Added dedicated `ProbabilityCalibrator` supporting Platt scaling, Isotonic regression, ECE, and Brier Score estimation.
- Integrated `SelectiveAbstentionGate` with `INSUFFICIENT_EVIDENCE` and `ABSTAIN` decision criteria.

### 2. Router & Interface Synchronization
- Fixed `overall_h_score` and `evidence_items` property access in `backend/app/modules/chat/router.py`.
- Preserved structured error semantics (`status="FAILED"`, `h_score=None`, `risk_level=None`) on unhandled exceptions.

### 3. Evaluation & Reproducibility
- Created `backend/evaluation/research_questions.yaml` registering hypotheses RQ1 through RQ7.
- Developed `backend/evaluation/run_comprehensive_research_evaluation.py` evaluating ablations A1-A12, baselines, cross-domain, cross-model transfer, and bootstrap 95% CIs.
- Generated `experiment_manifest.json` locking environment parameters, random seeds, and benchmark SHA-256 hash.
- Created complete documentation suite in `docs/`.
