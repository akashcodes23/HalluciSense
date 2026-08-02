"""Phase 6M — Configuration, Directory Paths, Schema Definitions, and Candidate Subsets.

Defines strict read-only paths to Phase 6I, 6K, and 6L artifacts, locked feature names,
the full 19-feature hybrid schema, and predefined candidate hybrid feature subsets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# Directory Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
EVALUATION_RESULTS_DIR: Path = BASE_DIR / "evaluation_results"
PHASE6I_DIR: Path = EVALUATION_RESULTS_DIR / "phase6i"
PHASE6K_DIR: Path = EVALUATION_RESULTS_DIR / "phase6k"
PHASE6L_DIR: Path = EVALUATION_RESULTS_DIR / "phase6l"
PHASE6M_DIR: Path = EVALUATION_RESULTS_DIR / "phase6m"
PHASE6M_FIGURES_DIR: Path = PHASE6M_DIR / "figures"
PHASE6M_FINAL_MODEL_DIR: Path = PHASE6M_DIR / "final_model"

# Pillar-1 Frozen Model Artifacts
PILLAR1_MODEL_DIR: Path = PHASE6K_DIR / "final_model"
PILLAR1_SCALER_PATH: Path = PILLAR1_MODEL_DIR / "robust_scaler.joblib"
PILLAR1_CLASSIFIER_PATH: Path = PILLAR1_MODEL_DIR / "pillar1_logistic_model.joblib"

# Pillar-2 Frozen Model Artifacts
PILLAR2_MODEL_DIR: Path = PHASE6L_DIR / "final_model"
PILLAR2_SCALER_PATH: Path = PILLAR2_MODEL_DIR / "preprocessing.joblib"
PILLAR2_CLASSIFIER_PATH: Path = PILLAR2_MODEL_DIR / "classifier.joblib"

# Dataset Paths
DEV_PHASE6I_PATH: Path = PHASE6I_DIR / "claim_evidence_features_development.jsonl"
VAL_PHASE6I_PATH: Path = PHASE6I_DIR / "claim_evidence_features_validation.jsonl"
DEV_PHASE6L_PATH: Path = PHASE6L_DIR / "structural_features_full_dev.jsonl"
VAL_PHASE6L_PATH: Path = PHASE6L_DIR / "structural_features_full_val.jsonl"

RANDOM_STATE: int = 42
EPSILON: float = 1e-6

# Locked Pillar-1 Features (5)
PILLAR1_LOCKED_FEATURES: List[str] = [
    "mean_entailment",
    "max_entailment",
    "mean_contradiction",
    "min_support_margin",
    "num_claims",
]

# Locked Pillar-2 Features (5)
PILLAR2_LOCKED_FEATURES: List[str] = [
    "max_pairwise_contradiction",
    "mean_pairwise_contradiction",
    "max_pairwise_similarity",
    "fraction_contradictory_pairs",
    "num_claims",
]

# Complete 19-Feature Hybrid Schema (Ordered)
HYBRID_FEATURE_SCHEMA: List[str] = [
    # Pillar-1 Features (5)
    "p1_mean_entailment",
    "p1_max_entailment",
    "p1_mean_contradiction",
    "p1_min_support_margin",
    "p1_num_claims",

    # Pillar-2 Features (5)
    "p2_max_pairwise_contradiction",
    "p2_mean_pairwise_contradiction",
    "p2_max_pairwise_similarity",
    "p2_fraction_contradictory_pairs",
    "p2_num_claims",

    # Probability Features (4)
    "prob_p1",
    "prob_p2",
    "logit_p1",
    "logit_p2",

    # Agreement & Disagreement Features (5)
    "prob_disagreement_abs",
    "prob_mean",
    "prob_max",
    "prob_min",
    "prob_ratio",
]

# Feature Families Mapping
FEATURE_FAMILIES: Dict[str, List[str]] = {
    "evidence_grounding": [
        "p1_mean_entailment",
        "p1_max_entailment",
        "p1_mean_contradiction",
        "p1_min_support_margin",
    ],
    "structural_consistency": [
        "p2_max_pairwise_contradiction",
        "p2_mean_pairwise_contradiction",
        "p2_max_pairwise_similarity",
        "p2_fraction_contradictory_pairs",
    ],
    "probability_signals": [
        "prob_p1",
        "prob_p2",
        "logit_p1",
        "logit_p2",
        "prob_disagreement_abs",
        "prob_mean",
        "prob_max",
        "prob_min",
        "prob_ratio",
    ],
    "response_controls": [
        "p1_num_claims",
        "p2_num_claims",
    ],
}

# Candidate Feature Subsets
CANDIDATE_SUBSETS: Dict[str, List[str]] = {
    "SET_A_FULL_HYBRID": HYBRID_FEATURE_SCHEMA,
    "SET_B_LATE_FUSION": [
        "prob_p1",
        "prob_p2",
        "logit_p1",
        "logit_p2",
    ],
    "SET_C_MID_LEVEL_LOCKED": [
        "p1_mean_entailment",
        "p1_max_entailment",
        "p1_mean_contradiction",
        "p1_min_support_margin",
        "p1_num_claims",
        "p2_max_pairwise_contradiction",
        "p2_mean_pairwise_contradiction",
        "p2_max_pairwise_similarity",
        "p2_fraction_contradictory_pairs",
        "p2_num_claims",
    ],
    "SET_D_COMPACT_HYBRID": [
        "prob_p1",
        "prob_p2",
        "prob_disagreement_abs",
        "p1_mean_entailment",
        "p1_min_support_margin",
        "p2_max_pairwise_contradiction",
        "p1_num_claims",
    ],
    "SET_E_META_SIGNALS_ONLY": [
        "prob_p1",
        "prob_p2",
        "logit_p1",
        "logit_p2",
        "prob_disagreement_abs",
        "prob_mean",
        "prob_ratio",
    ],
    "SET_F_EVIDENCE_PLUS_CONTRADICTION": [
        "p1_mean_entailment",
        "p1_max_entailment",
        "p1_min_support_margin",
        "p2_max_pairwise_contradiction",
        "p2_mean_pairwise_contradiction",
        "p1_num_claims",
    ],
}
