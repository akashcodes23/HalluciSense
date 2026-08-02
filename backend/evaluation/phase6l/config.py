"""Phase 6L — Configuration & Constants for Pillar-2 Structural Consistency.

Defines directory paths, cache filenames, exact 24-feature schema, thresholds, and default random seeds.
Strictly read-only with respect to historical Phase 6I/6J/6K artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

# Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
EVALUATION_RESULTS_DIR: Path = BASE_DIR / "evaluation_results"
PHASE6I_DIR: Path = EVALUATION_RESULTS_DIR / "phase6i"
PHASE6K_DIR: Path = EVALUATION_RESULTS_DIR / "phase6k"
PHASE6L_DIR: Path = EVALUATION_RESULTS_DIR / "phase6l"
PHASE6L_CACHE_DIR: Path = PHASE6L_DIR / "cache"
PHASE6L_FIGURES_DIR: Path = PHASE6L_DIR / "figures"

DEV_FEATURES_JSONL: Path = PHASE6I_DIR / "claim_evidence_features_development.jsonl"
VAL_FEATURES_JSONL: Path = PHASE6I_DIR / "claim_evidence_features_validation.jsonl"

RANDOM_STATE: int = 42

# Phase 6L.1A & 6L.1B Defaults
SUBSET_SIZE: int = 1000
DEFAULT_NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
SIMILARITY_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Threshold Configuration
TAU_CONTRADICTION: float = 0.50  # tau_C for contradiction pair classification
TAU_SUPPORT: float = 0.70        # tau_E for mutual support classification
TAU_SIMILARITY_DUPLICATE: float = 0.85 # tau_S for near-duplicate claim classification

FEATURE_SCHEMA_VERSION: str = "6L.1B.0"

# Exact 24-Feature Schema (Ordered)
STRUCTURAL_FEATURE_COLUMNS: List[str] = [
    # Family A — Pairwise Claim Contradiction
    "mean_pairwise_contradiction",
    "max_pairwise_contradiction",
    "p95_pairwise_contradiction",
    "fraction_contradictory_pairs",
    "contradiction_pair_count",

    # Family B — Claim Support / Agreement
    "mean_pairwise_entailment",
    "max_pairwise_entailment",
    "fraction_mutually_supportive_pairs",

    # Family C — Semantic Redundancy
    "mean_pairwise_similarity",
    "max_pairwise_similarity",
    "near_duplicate_claim_fraction",

    # Family D — Entity Consistency
    "entity_conflict_count",
    "entity_conflict_ratio",
    "entity_attribute_disagreement_score",

    # Family E — Numerical Consistency
    "numeric_conflict_count",
    "numeric_conflict_ratio",
    "max_numeric_disagreement",

    # Family F — Temporal Consistency
    "temporal_conflict_count",
    "timeline_order_violation_score",

    # Family G — Claim Graph Topology
    "contradiction_graph_density",
    "max_contradiction_degree",
    "largest_contradictory_component_ratio",

    # Family H — Response Structural Controls
    "num_claims",
    "claim_length_variance",
]
