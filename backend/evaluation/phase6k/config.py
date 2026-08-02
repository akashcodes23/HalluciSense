"""Phase 6K — Experiment Configuration and Feasibility Rules.

Defines global constants, paths, feasibility thresholds, model hyperparameters,
and random seeds for Phase 6K Stable Feature Selection & Model Recovery.

This module is analysis-only and read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# =========================================================
# PATHS
# =========================================================

PHASE6I_DIR: Path = Path("evaluation_results/phase6i")
PHASE6K_DIR: Path = Path("evaluation_results/phase6k")
PHASE6K_FIGURES_DIR: Path = Path("evaluation_results/phase6k/figures")

DEV_CACHE_FILENAME: str = "claim_evidence_features_development.jsonl"
VAL_CACHE_FILENAME: str = "claim_evidence_features_validation.jsonl"


# =========================================================
# FEATURE DEFINITIONS
# =========================================================

FEATURE_COLUMNS: List[str] = [
    "mean_entailment",
    "max_entailment",
    "mean_contradiction",
    "max_contradiction",
    "mean_support_margin",
    "min_support_margin",
    "fraction_supported",
    "fraction_contradicted",
    "fraction_unsupported",
    "num_claims",
]

TARGET_COLUMN: str = "ground_truth"


# =========================================================
# EXPERIMENT CONFIGURATION DATACLASS
# =========================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment configuration for Phase 6K execution."""

    seed: int = 42
    n_cv_folds: int = 5
    collinearity_threshold: float = 0.90
    max_condition_number: float = 1e4
    min_val_mcc: float = 0.20
    min_val_recall: float = 0.50
    min_val_specificity: float = 0.50

    # Scaler strategies to evaluate
    scalers: List[str] = field(
        default_factory=lambda: [
            "StandardScaler",
            "RobustScaler",
            "QuantileTransformer",
            "PowerTransformer",
        ]
    )

    # Classifiers to evaluate
    classifiers: List[str] = field(
        default_factory=lambda: [
            "LogisticRegression",
            "CalibratedSGD",
            "GaussianNB",
        ]
    )
