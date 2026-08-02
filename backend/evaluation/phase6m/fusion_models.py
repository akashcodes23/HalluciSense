"""Phase 6M — Candidate Model Factories & Preprocessors for Hybrid Fusion.

Defines candidate fusion algorithms, preprocessing choices, and model instantiators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

from evaluation.phase6m.config import CANDIDATE_SUBSETS, RANDOM_STATE


def get_preprocessor(scaler_type: Optional[str]) -> Any:
    """Instantiate requested scaling preprocessor."""
    if scaler_type == "StandardScaler":
        return StandardScaler()
    elif scaler_type == "RobustScaler":
        return RobustScaler()
    elif scaler_type is None or scaler_type == "None":
        return None
    else:
        raise ValueError(f"Unknown scaler type: '{scaler_type}'")


def get_candidate_configs() -> Dict[str, Dict[str, Any]]:
    """Return nominated hybrid candidate configurations."""
    candidates = {
        "Candidate 1": {
            "name": "Candidate 1 (SET_B + StandardScaler + LogisticRegression)",
            "set_key": "SET_B_LATE_FUSION",
            "scaler": "StandardScaler",
            "clf_factory": lambda: LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=RANDOM_STATE, max_iter=1000),
        },
        "Candidate 2": {
            "name": "Candidate 2 (SET_C + StandardScaler + LogisticRegression)",
            "set_key": "SET_C_MID_LEVEL_LOCKED",
            "scaler": "StandardScaler",
            "clf_factory": lambda: LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=RANDOM_STATE, max_iter=1000),
        },
        "Candidate 3": {
            "name": "Candidate 3 (SET_D + StandardScaler + RandomForest)",
            "set_key": "SET_D_COMPACT_HYBRID",
            "scaler": "StandardScaler",
            "clf_factory": lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
        },
        "Candidate 4": {
            "name": "Candidate 4 (SET_A + StandardScaler + RandomForest)",
            "set_key": "SET_A_FULL_HYBRID",
            "scaler": "StandardScaler",
            "clf_factory": lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
        },
        "Candidate 5": {
            "name": "Candidate 5 (SET_A + RobustScaler + HistGradientBoosting)",
            "set_key": "SET_A_FULL_HYBRID",
            "scaler": "RobustScaler",
            "clf_factory": lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=RANDOM_STATE),
        },
        "Candidate 6": {
            "name": "Candidate 6 (SET_A + StandardScaler + ExtraTrees)",
            "set_key": "SET_A_FULL_HYBRID",
            "scaler": "StandardScaler",
            "clf_factory": lambda: ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
        },
    }
    return candidates
