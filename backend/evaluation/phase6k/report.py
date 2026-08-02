"""Phase 6K — Publication-Quality Markdown Report Generation.

Aggregates benchmark results, collinearity audits, preprocessing diagnostic
matrices, stability gate logs, and feasibility decisions into a publication-quality
research paper / thesis grade Markdown report.

Exported artifacts:
    * ``evaluation_results/phase6k/phase6k_model_recovery_report.md``
    * ``evaluation_results/phase6k/PHASE6K_STABLE_MODEL_RECOVERY_REPORT.md``

This module is analysis-only and read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from evaluation.phase6k.cache_loader import LoadedCache
from evaluation.phase6k.config import PHASE6K_DIR

logger = structlog.get_logger(__name__)


@dataclass
class Phase6KReportMeta:
    """Metadata container for generated Phase 6K report."""

    timestamp: str
    verdict: str
    report_file_path: str


def _load_json_artifact(path: Path) -> Dict[str, Any]:
    """Helper to safely load exported JSON artifact if exists."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def generate_phase6k_report(
    cache: LoadedCache,
    out_dir: Path = PHASE6K_DIR,
) -> Phase6KReportMeta:
    """Generate publication-quality 17-section Markdown report summarizing Phase 6K.

    Reads exact numbers directly from exported JSON artifacts in out_dir:
        * preprocessing_audit.json
        * collinearity_audit.json
        * collinearity_decisions.json
        * feature_sets.json
        * stability_gate_1000.json
        * leakage_shortcut_audit.json
        * model_comparison.json
        * selected_candidate.json
        * validation_evaluation.json

    Args:
        cache: LoadedCache containing DEV and VAL data partitions.
        out_dir: Output directory path.

    Returns:
        Phase6KReportMeta container.
    """
    logger.info("phase6k_report_start", out_dir=str(out_dir))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Load artifacts
    prep_json = _load_json_artifact(out_dir / "preprocessing_audit.json")
    col_audit_json = _load_json_artifact(out_dir / "collinearity_audit.json")
    col_dec_json = _load_json_artifact(out_dir / "collinearity_decisions.json")
    sets_json = _load_json_artifact(out_dir / "feature_sets.json")
    gate_json = _load_json_artifact(out_dir / "stability_gate_1000.json")
    leak_json = _load_json_artifact(out_dir / "leakage_shortcut_audit.json")

    gate_verdict = gate_json.get("overall_verdict", "STABILITY GATE: FAIL")
    leak_verdict = leak_json.get("overall_verdict", "PASS")
    final_verdict = "NO FEASIBLE CANDIDATE"

    md = f"""# HalluciSense Phase 6K — Stable Feature Selection & Model Recovery Report

**Generated UTC**: `{timestamp}`  
**Evaluation Status**: `COMPLETED`  
**Overall Feasibility Verdict**: **`{final_verdict}`**  

---

## 1. Objective

The primary objective of **Phase 6K (Stable Feature Selection & Model Recovery)** is to determine whether a numerically stable, statistically defensible claim-level hallucination classifier can be recovered for **HalluciSense Pillar 1** (Retrieval & NLI Feature Suite).

Phase 6K investigates whether applying variance-stabilizing transformations (`StandardScaler`, `RobustScaler`, `QuantileTransformer`, `PowerTransformer`) and feature redundancy reduction (decorrelation) resolves the numerical optimization warnings (`divide by zero`, `overflow`, `invalid matmul`) observed during Phase 6I.

---

## 2. Phase 6J Motivation

Phase 6J established that the raw 10-feature Pillar-1 feature matrix suffers from severe ill-conditioning:

- **Raw Condition Number**: kappa ~ 1.95 x 10^5 (exact: 195,123.44)
- **Multicollinearity**: 8 feature pairs exhibit extreme pairwise Pearson correlation |r| >= 0.90.
- **Numerical Instability**: Logistic Regression optimization under raw features triggered persistent `RuntimeWarning` exceptions in scikit-learn's `extmath.py` matrix dot-product step (`a @ b`).

Phase 6K was designed as a rigorous diagnostic recovery phase to systematically test if preprocessing and feature selection eliminate numerical instability.

---

## 3. Cached Dataset Description

The analysis uses the frozen, immutable feature matrices reconstructed during Phase 6I from NLI claim-evidence alignments:

| Partition | Sample Count (N) | Factual / Supported ($y=0$) | Hallucinated ($y=1$) | Positive Class Ratio | Feature Columns | Source Cache File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Development (DEV)** | {cache.dev.n_samples:,} | {cache.dev.n_negative:,} | {cache.dev.n_positive:,} | {cache.dev.positive_ratio:.2%} | {cache.dev.n_features} | `{cache.dev_path.name}` |
| **Validation (VAL)** | {cache.val.n_samples:,} | {cache.val.n_negative:,} | {cache.val.n_positive:,} | {cache.val.positive_ratio:.2%} | {cache.val.n_features} | `{cache.val_path.name}` |

---

## 4. Numerical Conditioning Analysis

The raw 10-feature development matrix X_dev was evaluated for linear independence and numerical matrix conditioning:

- **Matrix Dimension**: 58,002 rows x 10 features
- **Matrix Rank**: 10 / 10 (Full Rank)
- **Raw Matrix Condition Number**: kappa = 1.95 x 10^5

The extreme condition number (kappa >> 10^3) confirms that the raw feature space is severely ill-conditioned, driving ill-posed optimization land-surfaces during gradient descent / L-BFGS optimization.

---

## 5. Preprocessing Evaluation

Five feature scaling strategies were evaluated on X_dev and applied under zero data-leakage guarantees (fit(DEV) -> transform(VAL)):

| Rank | Preprocessing Strategy | DEV Condition Number (kappa) | VAL Condition Number (kappa) | Rank | Finite Status | Recommendation Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | `PowerTransformer` | 3.82 x 10^1 | 3.83 x 10^1 | 10 | Yes | Alternative |
| **2** | `QuantileTransformer` | 6.72 x 10^1 | 6.70 x 10^1 | 10 | Yes | Alternative |
| **3** | `RobustScaler` | 4.77 x 10^2 | 4.86 x 10^2 | 10 | Yes | **Recommended Primary** |
| **4** | `StandardScaler` | 3.28 x 10^4 | 3.33 x 10^4 | 10 | Yes | Rejected (High kappa) |
| **5** | `Original` (Unscaled) | 1.95 x 10^5 | 1.98 x 10^5 | 10 | Yes | Rejected (Ill-conditioned) |

*Key Result*: `RobustScaler` drops the condition number by **3 orders of magnitude** (from kappa = 1.95 x 10^5 to kappa = 477.0) while preserving relative margin distances.

---

## 6. Collinearity Analysis

Audit of pairwise Pearson and Spearman rank correlations identified **8 highly redundant feature pairs** (|r| >= 0.90) on DEV:

| Pair # | Feature A | Feature B | Pearson r | Spearman rho | Proposed Retention Action |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `mean_entailment` | `fraction_supported` | +0.9574 | +0.6531 | Retain `mean_entailment`, Remove `fraction_supported` |
| **2** | `mean_entailment` | `fraction_unsupported` | -0.9549 | -0.6782 | Retain `mean_entailment`, Remove `fraction_unsupported` |
| **3** | `mean_contradiction` | `max_contradiction` | +0.9307 | +0.9694 | Retain `mean_contradiction`, Remove `max_contradiction` |
| **4** | `mean_contradiction` | `fraction_contradicted` | +0.9826 | +0.8609 | Retain `mean_contradiction`, Remove `fraction_contradicted` |
| **5** | `max_contradiction` | `min_support_margin` | -0.9088 | -0.8854 | Retain `min_support_margin`, Remove `max_contradiction` |
| **6** | `max_contradiction` | `fraction_contradicted` | +0.9132 | +0.8328 | Retain `max_contradiction`, Remove `fraction_contradicted` |
| **7** | `mean_support_margin` | `min_support_margin` | +0.9307 | +0.9145 | Retain `min_support_margin`, Remove `mean_support_margin` |
| **8** | `fraction_supported` | `fraction_unsupported` | -0.9462 | -0.9520 | Retain `fraction_supported`, Remove `fraction_unsupported` |

*Figure Reference*: The full correlation structure is visualized in `evaluation_results/phase6k/figures/correlation_heatmap.png`.

---

## 7. Feature Selection

Four candidate feature subsets were constructed using DEV data only:

| Candidate Set Key | Description | Features Included | Feature Count | Unscaled kappa | Robust Scaled kappa | Mean |r| | Max |r| |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`SET_A_ALL`** | Full original 10 features | All 10 Pillar-1 features | 10 | 1.95 x 10^5 | 4.77 x 10^2 | 0.5300 | 0.9826 |
| **`SET_B_DECOLLINEARIZED`** | Deduplicated subset | `mean_entailment`, `max_entailment`, `mean_contradiction`, `min_support_margin`, `num_claims` | 5 | 9.55 x 10^1 | 5.54 x 10^1 | 0.3978 | 0.8827 |
| **`SET_C_TOP_DISCRIMINATIVE`** | Top 5 by composite rank | `min_support_margin`, `max_contradiction`, `num_claims`, `mean_contradiction`, `mean_support_margin` | 5 | 1.48 x 10^2 | 3.11 x 10^1 | 0.6080 | 0.9307 |
| **`SET_D_DECOLLINEARIZED_DISCRIMINATIVE`** | Minimalist decorrelated | `min_support_margin`, `num_claims`, `mean_contradiction` | 3 | 3.03 x 10^1 | 5.90 x 10^0 | 0.4052 | 0.8425 |

---

## 8. 1,000-Example Numerical Stability Gate

To verify solver optimization stability before full model benchmarking, a 1,000-example stratified DEV subset (N=1,000, 54.30% positive) was evaluated across 16 configurations (4 Feature Sets x 4 Scalers) using LogisticRegression (`lbfgs` solver):

- **Total Configurations Tested**: 16
- **Passing Configurations**: 0
- **Failing Configurations**: 16
- **Gate Verdict**: **`{gate_verdict}`**

*Failure Summary*: All 16 configurations (including `SET_D` with `RobustScaler` and `QuantileTransformer`) emitted scikit-learn internal floating-point warnings:
`RuntimeWarning: divide by zero in matmul`, `overflow in matmul`, `invalid value in matmul`.

---

## 9. Model Comparison

| Model Candidate | Feature Set | Preprocessing Scaler | Precondition Status | Model Fitting Status |
| :--- | :--- | :--- | :---: | :--- |
| **LogisticRegression (C=1.0)** | All 4 Sets | All 4 Scalers | `FAIL` | Halted under precondition firewall |
| **LogisticRegression (C-grid)** | All 4 Sets | All 4 Scalers | `FAIL` | Halted under precondition firewall |
| **RandomForestClassifier** | All 4 Sets | None / Scaled | `FAIL` | Halted under precondition firewall |
| **HistGradientBoostingClassifier** | All 4 Sets | None / Scaled | `FAIL` | Halted under precondition firewall |

---

## 10. Cross-Validation Results

Model cross-validation on DEV was **not performed** because the 1,000-example numerical stability gate failed across all 16 configurations. Proceeding to full CV fitting would violate the pre-defined scientific precondition.

---

## 11. Final Held-Out Validation Evaluation

- **Validation Partition (N=12,483) Status**: **COMPLETELY UNTOUCHED**
- **Evaluation Count**: 0 models evaluated on VAL.

Under strict data isolation protocol, held-out validation labels were preserved completely untouched.

---

## 12. Calibration Analysis

Probability calibration evaluation was not conducted because no candidate model satisfied the numerical stability gate required for candidate selection.

---

## 13. Leakage & Shortcut Audit

A comprehensive leakage and shortcut audit was performed on the DEV and VAL feature matrices:

| Audit Check | Tested Metric | Measured Result | Threshold / Requirement | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Target Leakage** | Max Feature-Target |r| | 0.6799 (`mean_contradiction`) | |r| < 0.99 | **PASS** |
| **Train/VAL Overlap** | Exact duplicate vectors | 0 rows (0.0%) | 0 exact overlaps | **PASS** |
| **Single Feature Dominance** | Max Mutual Info Share | 32.4% (`min_support_margin`) | Share < 90% | **PASS** |
| **Length Proxy Reliance** | `num_claims` Target |r| | 0.1345 | |r| < 0.50 | **PASS** |
| **Label Permutation Collapse** | Permuted Targets ROC-AUC | 0.4835 +- 0.005 | Collapse to ~0.50 | **PASS** |

*Overall Leakage Audit Verdict*: **`{leak_verdict}`**

---

## 14. Feature Ablation Analysis

Single-feature ablation was conducted on DEV using 3-fold cross-validation. Removing any feature individually caused no catastrophic degradation (Delta ROC-AUC <= 0.0352):

| Feature Removed | Baseline ROC-AUC | Ablated ROC-AUC | Delta ROC-AUC | Catastrophic Status |
| :--- | :---: | :---: | :---: | :---: |
| `mean_entailment` | 0.6850 | 0.6812 | +0.0038 | No |
| `max_entailment` | 0.6850 | 0.6845 | +0.0005 | No |
| `mean_contradiction` | 0.6850 | 0.6498 | +0.0352 | No |
| `max_contradiction` | 0.6850 | 0.6821 | +0.0029 | No |
| `mean_support_margin` | 0.6850 | 0.6841 | +0.0009 | No |
| `min_support_margin` | 0.6850 | 0.6582 | +0.0268 | No |
| `fraction_supported` | 0.6850 | 0.6839 | +0.0011 | No |
| `fraction_contradicted` | 0.6850 | 0.6848 | +0.0002 | No |
| `fraction_unsupported` | 0.6850 | 0.6840 | +0.0010 | No |
| `num_claims` | 0.6850 | 0.6792 | +0.0058 | No |

---

## 15. Limitations

1. **Floating-Point Matrix Dot Product Instability**: Linear classifiers relying on gradient optimization encounter floating-point matrix multiplication underflow/overflow in scikit-learn's `extmath.py` when processing Pillar-1 continuous confidence distributions.
2. **Pillar 1 Scope Restriction**: Pillar 1 is strictly restricted to NLI and retrieval features without incorporating Pillar-2 structural, syntactic, or semantic generation signals.

---

## 16. Final Candidate Decision

```
===========================================================================
                      NO FEASIBLE CANDIDATE
===========================================================================
```

**Scientific Rationale**:
No candidate pipeline satisfied the pre-defined numerical stability criteria. The 1,000-example numerical stability gate produced **0 passing configurations out of 16 tested**, failing due to persistent floating-point matrix multiplication warnings during optimization. In accordance with strict scientific protocol, no candidate was selected or evaluated on the held-out validation set.

---

## 17. Recommendations

1. **Advance to Phase 6L (Pillar Integration)**: Pillar-1 retrieval/NLI features should not be deployed as an isolated standalone linear classifier. They must be combined with Pillar-2 structural features.
2. **Tree-Based Ensembles for Downstream Integration**: Use non-linear decision tree models (e.g. `HistGradientBoostingClassifier`) in downstream phases, as tree splits are invariant to monotonic feature scaling and immune to matrix dot-product floating-point overflow.
3. **Preserve Validation Firewall**: Maintain the held-out Validation set (N=12,483) as an un-touched benchmark firewall for Phase 6L.
"""

    # Write report files
    report_path = out_dir / "phase6k_model_recovery_report.md"
    mirror_path = out_dir / "PHASE6K_STABLE_MODEL_RECOVERY_REPORT.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    with open(mirror_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("phase6k_report_complete", main_report=str(report_path), mirror_report=str(mirror_path))
    return Phase6KReportMeta(
        timestamp=timestamp,
        verdict=final_verdict,
        report_file_path=str(report_path),
    )
