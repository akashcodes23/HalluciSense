"""Phase 6K — Model Factory Definitions for Model Recovery Experiment.

Provides factory functions to instantiate scikit-learn classifiers under
controlled deterministic hyperparameters (random_state=42):

    1. LogisticRegression (default C=1.0)
    2. Regularized LogisticRegression across C grid [0.001, 0.01, 0.1, 1.0, 10.0]
    3. RandomForestClassifier (n_estimators=100, max_depth=5, random_state=42)
    4. HistGradientBoostingClassifier (max_depth=5, random_state=42)

No external ML libraries (e.g. XGBoost, LightGBM, CatBoost) are introduced.

This module is analysis-only and read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import structlog
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

logger = structlog.get_logger(__name__)

C_GRID: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0]


@dataclass
class ModelSpec:
    """Specification metadata for a classifier model."""

    name: str
    model_class: str
    hyperparameters: Dict[str, Any]


def get_model_factory(
    model_name: str,
    random_state: int = 42,
) -> Tuple[BaseEstimator, ModelSpec]:
    """Instantiate a classifier model and its ModelSpec metadata.

    Args:
        model_name: Identifier of classifier model.
        random_state: Deterministic random seed.

    Returns:
        Tuple of (unfitted_sklearn_estimator, ModelSpec).

    Raises:
        ValueError: If model_name is not supported.
    """
    logger.info("phase6k_get_model_factory", model=model_name, seed=random_state)

    if model_name == "LogisticRegression":
        hp = {"solver": "lbfgs", "max_iter": 1000, "random_state": random_state, "C": 1.0}
        model = LogisticRegression(**hp)
        spec = ModelSpec(name=model_name, model_class="LogisticRegression", hyperparameters=hp)
        return model, spec

    if model_name.startswith("LogisticRegression_C"):
        try:
            c_val = float(model_name.split("_C")[1])
        except Exception:
            c_val = 1.0
        hp = {"solver": "lbfgs", "max_iter": 1000, "random_state": random_state, "C": c_val}
        model = LogisticRegression(**hp)
        spec = ModelSpec(name=model_name, model_class="LogisticRegression", hyperparameters=hp)
        return model, spec

    if model_name == "RandomForestClassifier":
        hp = {"n_estimators": 100, "max_depth": 5, "random_state": random_state}
        model = RandomForestClassifier(**hp)
        spec = ModelSpec(name=model_name, model_class="RandomForestClassifier", hyperparameters=hp)
        return model, spec

    if model_name == "HistGradientBoostingClassifier":
        hp = {"max_depth": 5, "random_state": random_state}
        model = HistGradientBoostingClassifier(**hp)
        spec = ModelSpec(name=model_name, model_class="HistGradientBoostingClassifier", hyperparameters=hp)
        return model, spec

    raise ValueError(f"Unsupported model_name: {model_name!r}")
