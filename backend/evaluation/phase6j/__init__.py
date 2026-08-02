"""HalluciSense Phase 6J — Numerical Stability & Feature Validation.

Post-hoc analysis of cached Phase 6I feature matrices.
Read-only: never modifies Phase 6I outputs.

Modules:
    statistics   — Descriptive statistics per feature.
    distributions — Distribution shape analysis (skew, kurtosis, normality).
    scaling      — Scaling diagnostics (range, outliers, dynamic range).
    separation   — Class-conditional separation metrics.
    stability    — Numerical stability checks (NaN, Inf, condition number).
    report       — Consolidated Markdown report generation.
    run_phase6j  — Orchestrator.
"""
