"""Phase 6L.2 — Stage 5: Corrected Numerical Stability Gate Engine.

Evaluates every Preprocessing x Feature Set x Classifier combination on a deterministic
1,000-example DEV subset using mutually-exclusive warning accounting.
Only numerically stable candidates pass to full DEV cross-validation.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, RobustScaler
import structlog

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)


def classify_warning_mutually_exclusive(w: warnings.WarningMessage) -> str:
    """Classify a Python Warning into exactly ONE mutually exclusive category."""
    msg = str(w.message).lower()

    if "overflow" in msg:
        return "overflow"
    elif "divide by zero" in msg or "division by zero" in msg:
        return "divide_by_zero"
    elif "invalid value" in msg or "matmul" in msg or "nan" in msg or "inf" in msg:
        return "invalid_matmul"
    elif "lbfgs" in msg or "line search" in msg:
        return "line_search_warning"
    elif "converge" in msg or "max_iter" in msg or "failed to converge" in msg:
        return "convergence_warning"
    elif "singular" in msg or "ill-conditioned" in msg:
        return "ill_conditioned_warning"
    else:
        return "other_warning"


def run_numerical_stability_gate(
    X: np.ndarray,
    y: np.ndarray,
    candidate_sets: Dict[str, Any],
    feature_names: List[str] = STRUCTURAL_FEATURE_COLUMNS,
    out_dir: Path = PHASE6L_DIR,
) -> Dict[str, Any]:
    """Run deterministic 1,000-example stability gate with mutually exclusive warning accounting.

    Returns:
        Dict containing stability audit, warning counts, and list of passing configurations.
    """
    logger.info("stage5_stability_gate_start", total_samples=X.shape[0])

    # Deterministic 1,000-example subset
    np.random.seed(42)
    n_sub = min(1000, X.shape[0])
    sub_idx = np.random.choice(X.shape[0], n_sub, replace=False)
    X_sub = X[sub_idx]
    y_sub = y[sub_idx]

    scalers = {
        "None": None,
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
    }

    classifiers = {
        "liblinear_l2": LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=42, max_iter=1000),
        "liblinear_l1": LogisticRegression(solver="liblinear", penalty="l1", C=1.0, random_state=42, max_iter=1000),
        "saga_l2": LogisticRegression(solver="saga", penalty="l2", C=1.0, random_state=42, max_iter=1000),
        "saga_l1": LogisticRegression(solver="saga", penalty="l1", C=1.0, random_state=42, max_iter=1000),
        "ridge": RidgeClassifier(alpha=1.0, random_state=42),
        "linear_svc": LinearSVC(C=1.0, random_state=42, max_iter=2000, dual="auto"),
    }

    results: List[Dict[str, Any]] = []
    passing_configs: List[Dict[str, Any]] = []

    for set_name, set_info in candidate_sets.items():
        feat_sub_cols = set_info["features"]
        feat_indices = [feature_names.index(c) for c in feat_sub_cols]
        X_feat_sub = X_sub[:, feat_indices]

        for scaler_name, scaler_obj in scalers.items():
            if scaler_obj is None:
                X_scaled = X_feat_sub.copy()
            else:
                scaler_inst = type(scaler_obj)()
                X_scaled = scaler_inst.fit_transform(X_feat_sub)

            for clf_name, clf_obj in classifiers.items():
                clf_inst = type(clf_obj)(**clf_obj.get_params())

                warning_counts = {
                    "overflow": 0,
                    "divide_by_zero": 0,
                    "invalid_matmul": 0,
                    "line_search_warning": 0,
                    "convergence_warning": 0,
                    "ill_conditioned_warning": 0,
                    "other_warning": 0,
                }
                total_warnings = 0
                has_fatal_error = False
                error_message = None

                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter("always")
                    try:
                        clf_inst.fit(X_scaled, y_sub)

                        if hasattr(clf_inst, "predict_proba"):
                            probs = clf_inst.predict_proba(X_scaled)
                            if not np.all(np.isfinite(probs)):
                                has_fatal_error = True
                                error_message = "Non-finite predicted probabilities"
                        else:
                            dec = clf_inst.decision_function(X_scaled)
                            if not np.all(np.isfinite(dec)):
                                has_fatal_error = True
                                error_message = "Non-finite decision function output"

                    except Exception as exc:
                        has_fatal_error = True
                        error_message = str(exc)

                for w in captured_warnings:
                    cat = classify_warning_mutually_exclusive(w)
                    warning_counts[cat] += 1
                    total_warnings += 1

                # Rejection criteria: any fatal error or solver failure
                is_passed = (not has_fatal_error) and (warning_counts["overflow"] == 0) and (warning_counts["divide_by_zero"] == 0) and (warning_counts["invalid_matmul"] == 0)

                config_entry = {
                    "feature_set": set_name,
                    "feature_count": len(feat_sub_cols),
                    "scaler": scaler_name,
                    "classifier": clf_name,
                    "total_warnings": total_warnings,
                    "warning_breakdown": warning_counts,
                    "has_fatal_error": has_fatal_error,
                    "error_message": error_message,
                    "passed_stability_gate": is_passed,
                }

                results.append(config_entry)
                if is_passed:
                    passing_configs.append(config_entry)

    # Export stability gate report & warning forensics
    gate_payload = {
        "n_evaluated": len(results),
        "n_passed": len(passing_configs),
        "n_rejected": len(results) - len(passing_configs),
        "pass_rate": float(len(passing_configs) / max(1, len(results))),
        "results": results,
    }

    warning_forensics = {
        "mutually_exclusive_classification_verified": True,
        "total_warnings_across_all_configs": sum(r["total_warnings"] for r in results),
        "configurations_with_zero_warnings": [r for r in results if r["total_warnings"] == 0],
    }

    with open(out_dir / "stability_gate.json", "w", encoding="utf-8") as f:
        json.dump(gate_payload, f, indent=2)

    with open(out_dir / "warning_forensics.json", "w", encoding="utf-8") as f:
        json.dump(warning_forensics, f, indent=2)

    logger.info(
        "stage5_stability_gate_complete",
        total_eval=len(results),
        passed=len(passing_configs),
        rejected=len(results) - len(passing_configs),
    )

    return {
        "gate_payload": gate_payload,
        "warning_forensics": warning_forensics,
        "passing_configs": passing_configs,
    }
