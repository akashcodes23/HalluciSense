"""Phase 37 — Local Counterfactual Attribution Engine.

Implements one-feature-at-a-time counterfactual attribution against the frozen
19-feature HalluciSense Hybrid classifier (HistGradientBoostingClassifier).

Attribution Method: Local Counterfactual Attribution
=====================================================

For each feature i:
    a_i = P(H | X) - P(H | X_i)

where X_i is the original 19-dimensional feature vector X with feature i
replaced by its training-median value (sourced from RobustScaler.center_).

Interpretation:
    a_i > 0 → feature i locally increases hallucination probability
    a_i < 0 → feature i locally decreases hallucination probability
    a_i ≈ 0 → feature i has negligible local effect

Interaction Gap:
    Because the classifier is nonlinear (HistGradientBoostingClassifier),
    individual one-feature counterfactuals are NOT an additive decomposition.
    The residual is captured as:

    interaction_gap = [P(H | X) - P(H | X_baseline)] - Σ a_i

This is NOT SHAP. SHAP uses Shapley values which require marginalising over
all feature coalitions. This method uses one-feature-at-a-time counterfactuals
for computational efficiency and interpretability.

Scientific Guarantees:
    - Deterministic: same (X, artifacts) → identical attributions
    - Model-faithful: only calls the frozen production classifier
    - Non-destructive: never modifies classifier, scaler, threshold, or ordering
    - Traceable: baseline sourced from frozen RobustScaler.center_ (training medians)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical paths — resolved at import time so nothing is hardcoded
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # backend/


def _resolve_model_dir() -> Path:
    """Robustly resolve phase6m final_hybrid_model directory."""
    candidates = [
        _BASE_DIR / "evaluation_results" / "phase6m" / "final_hybrid_model",
        _BASE_DIR / "backend" / "evaluation_results" / "phase6m" / "final_hybrid_model",
        Path.cwd() / "evaluation_results" / "phase6m" / "final_hybrid_model",
        Path.cwd() / "backend" / "evaluation_results" / "phase6m" / "final_hybrid_model",
    ]
    for c in candidates:
        if c.exists():
            return c
    return _BASE_DIR / "evaluation_results" / "phase6m" / "final_hybrid_model"


# ---------------------------------------------------------------------------
# Feature schema — loaded from model_metadata.json (NOT hardcoded)
# ---------------------------------------------------------------------------

def _load_feature_schema(model_dir: Path) -> List[str]:
    """Load canonical 19-feature schema from model_metadata.json."""
    meta_path = model_dir / "model_metadata.json"
    schema_path = model_dir / "feature_schema.json"

    # Primary: model_metadata.json protocol.feature_schema
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            schema = meta.get("protocol", {}).get("feature_schema", [])
            if len(schema) == 19:
                logger.debug("feature_schema_loaded_from_model_metadata", count=len(schema))
                return schema
        except Exception as exc:
            logger.warning("feature_schema_load_failed_model_metadata", error=str(exc))

    # Fallback: feature_schema.json
    if schema_path.exists():
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            schema = data.get("feature_schema", [])
            if len(schema) == 19:
                logger.debug("feature_schema_loaded_from_feature_schema_json", count=len(schema))
                return schema
        except Exception as exc:
            logger.warning("feature_schema_load_failed_feature_schema_json", error=str(exc))

    raise RuntimeError(
        "Cannot load canonical 19-feature schema from model_metadata.json or feature_schema.json. "
        f"Checked: {meta_path}, {schema_path}"
    )


# ---------------------------------------------------------------------------
# Training-median baseline — from RobustScaler.center_
# ---------------------------------------------------------------------------

def _load_training_medians(preprocessing_path: Path) -> np.ndarray:
    """Extract training medians from frozen RobustScaler.center_ attribute.

    RobustScaler.center_ stores the median of each feature computed during
    fit() on the training set (N=58,002 development instances). These are the
    canonical training-distribution medians used as the counterfactual baseline.
    """
    from app.models.registry import safe_joblib_load  # lazy import to avoid circular

    scaler = safe_joblib_load(preprocessing_path)
    if not hasattr(scaler, "center_"):
        raise AttributeError(
            f"Loaded object from {preprocessing_path} does not have 'center_' attribute. "
            f"Expected sklearn.preprocessing.RobustScaler, got {type(scaler).__name__}."
        )

    medians = np.array(scaler.center_, dtype=np.float64)
    if len(medians) != 19:
        raise ValueError(
            f"RobustScaler.center_ has {len(medians)} values, expected 19. "
            "Preprocessing artifact may be corrupted."
        )
    logger.debug("training_medians_loaded_from_robust_scaler", n_features=len(medians))
    return medians


# ---------------------------------------------------------------------------
# Module-level cached state (loaded once per process)
# ---------------------------------------------------------------------------

_model_dir: Optional[Path] = None
_feature_schema: Optional[List[str]] = None
_training_medians: Optional[np.ndarray] = None


def _ensure_loaded() -> None:
    """Lazily load feature schema and training medians exactly once."""
    global _model_dir, _feature_schema, _training_medians

    if _feature_schema is not None and _training_medians is not None:
        return

    _model_dir = _resolve_model_dir()
    _feature_schema = _load_feature_schema(_model_dir)
    _training_medians = _load_training_medians(_model_dir / "preprocessing.joblib")
    logger.info(
        "local_attribution_engine_initialized",
        feature_count=len(_feature_schema),
        baseline_source="RobustScaler.center_",
        model_dir=str(_model_dir),
    )


def get_feature_schema() -> List[str]:
    """Return canonical 19-feature schema (loaded from model_metadata.json)."""
    _ensure_loaded()
    return list(_feature_schema)  # type: ignore[arg-type]


def get_training_medians() -> np.ndarray:
    """Return training-median baseline vector (from RobustScaler.center_)."""
    _ensure_loaded()
    return _training_medians.copy()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

EXPECTED_FEATURE_COUNT = 19


def validate_feature_vector(X_raw: np.ndarray) -> None:
    """Validate a raw 19-dimensional feature vector before attribution.

    Args:
        X_raw: numpy array of shape (1, 19) or (19,).

    Raises:
        ValueError: if the vector is invalid.
    """
    if X_raw is None:
        raise ValueError("Feature vector is None.")

    arr = np.atleast_1d(np.array(X_raw, dtype=np.float64))
    if arr.ndim == 2:
        arr = arr.flatten()

    if len(arr) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Feature vector must have exactly {EXPECTED_FEATURE_COUNT} elements. "
            f"Received {len(arr)}."
        )

    if not np.all(np.isfinite(arr)):
        nan_positions = [i for i, v in enumerate(arr) if math.isnan(v)]
        inf_positions = [i for i, v in enumerate(arr) if math.isinf(v)]
        raise ValueError(
            f"Feature vector contains non-finite values. "
            f"NaN at positions: {nan_positions}. Inf at positions: {inf_positions}."
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureAttribution:
    """Attribution result for a single feature."""
    feature_name: str
    index: int
    value: float          # actual value in this prediction
    baseline: float       # training-median baseline
    attribution: float    # a_i = P_original - P_i
    direction: str        # "hallucination_risk" | "protective" | "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "index": self.index,
            "value": round(self.value, 6),
            "baseline": round(self.baseline, 6),
            "attribution": round(self.attribution, 6),
            "direction": self.direction,
        }


@dataclass
class LocalAttributionResult:
    """Complete local counterfactual attribution result for one prediction."""
    method: str = "local_counterfactual_attribution"
    feature_count: int = 19
    baseline_type: str = "training_median_from_robust_scaler"
    original_probability: float = 0.0
    baseline_probability: float = 0.0
    threshold: float = 0.54
    decision_margin: float = 0.0
    interaction_gap: float = 0.0
    features: List[FeatureAttribution] = field(default_factory=list)
    top_hallucination_drivers: List[FeatureAttribution] = field(default_factory=list)
    top_protective_drivers: List[FeatureAttribution] = field(default_factory=list)
    inference_count: int = 0  # number of model evaluations performed

    def to_dict(self) -> Dict[str, Any]:
        interaction_gap_abs = abs(self.interaction_gap)
        if interaction_gap_abs > 0.01:
            gap_explanation = (
                f"The classifier is nonlinear (HistGradientBoostingClassifier), so "
                f"individual one-feature counterfactual contributions do not sum exactly to "
                f"the total prediction change. The residual interaction gap of "
                f"{self.interaction_gap:+.4f} captures nonlinear interaction effects "
                f"between features."
            )
        else:
            gap_explanation = (
                "Individual feature contributions approximately account for the total "
                f"prediction change (interaction gap = {self.interaction_gap:+.4f})."
            )

        return {
            "method": self.method,
            "feature_count": self.feature_count,
            "baseline_type": self.baseline_type,
            "original_probability": round(self.original_probability, 6),
            "baseline_probability": round(self.baseline_probability, 6),
            "threshold": self.threshold,
            "decision_margin": round(self.decision_margin, 6),
            "interaction_gap": round(self.interaction_gap, 6),
            "interaction_gap_explanation": gap_explanation,
            "scientific_caveat": (
                "Attributions describe the local behavior of the classifier for this specific input. "
                "They are not independent proof that a claim is true or false. "
                "The counterfactual effect of each feature is measured relative to its "
                "training-median baseline value."
            ),
            "features": [f.to_dict() for f in self.features],
            "top_hallucination_drivers": [f.to_dict() for f in self.top_hallucination_drivers],
            "top_protective_drivers": [f.to_dict() for f in self.top_protective_drivers],
            "inference_count": self.inference_count,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

_DIRECTION_THRESHOLD = 0.002  # minimum |attribution| to consider non-neutral


def _direction(attribution: float) -> str:
    if attribution > _DIRECTION_THRESHOLD:
        return "hallucination_risk"
    if attribution < -_DIRECTION_THRESHOLD:
        return "protective"
    return "neutral"


def compute_local_attribution(
    X_raw: np.ndarray,
    scaler: Any,
    clf: Any,
    threshold: float = 0.54,
    top_k: int = 5,
) -> LocalAttributionResult:
    """Compute local counterfactual attribution for a 19-feature prediction.

    Algorithm:
        1. Validate input vector (19 features, finite values).
        2. Compute P_original = clf.predict_proba(scaler.transform(X_raw))[0,1].
        3. Build X_baseline = training medians (from RobustScaler.center_).
        4. Compute P_baseline = clf.predict_proba(scaler.transform(X_baseline))[0,1].
        5. For each feature i (0..18):
               X_i = X_raw.copy(); X_i[0, i] = training_median[i]
               P_i = clf.predict_proba(scaler.transform(X_i))[0,1]
               a_i = P_original - P_i
        6. interaction_gap = (P_original - P_baseline) - sum(a_i)
        7. Sort features by attribution for top drivers.

    Args:
        X_raw:     Unscaled 1x19 raw feature array.
        scaler:    Frozen RobustScaler (must not be modified).
        clf:       Frozen HistGradientBoostingClassifier (must not be modified).
        threshold: Production decision threshold (default 0.54).
        top_k:     Number of top drivers to return in each direction.

    Returns:
        LocalAttributionResult with all attribution data.

    Raises:
        ValueError: if input vector is invalid.
        RuntimeError: if feature schema or training medians cannot be loaded.
    """
    _ensure_loaded()
    feature_schema = _feature_schema  # type: ignore[assignment]
    training_medians = _training_medians  # type: ignore[assignment]

    # Ensure shape (1, 19)
    X = np.atleast_2d(np.array(X_raw, dtype=np.float64))
    validate_feature_vector(X)
    if X.shape != (1, 19):
        X = X.reshape(1, 19)

    # ── Step 2: Original probability ──────────────────────────────────────
    X_scaled = scaler.transform(X)
    P_original = float(clf.predict_proba(X_scaled)[0, 1])
    inference_count = 1

    # ── Step 4: Baseline probability ──────────────────────────────────────
    X_baseline = training_medians.reshape(1, 19)
    X_baseline_scaled = scaler.transform(X_baseline)
    P_baseline = float(clf.predict_proba(X_baseline_scaled)[0, 1])
    inference_count += 1

    # ── Step 5: Per-feature counterfactual attributions ───────────────────
    attributions: List[FeatureAttribution] = []
    sum_attributions = 0.0

    for i, fname in enumerate(feature_schema):  # type: ignore[arg-type]
        X_i = X.copy()
        X_i[0, i] = training_medians[i]
        X_i_scaled = scaler.transform(X_i)
        P_i = float(clf.predict_proba(X_i_scaled)[0, 1])
        inference_count += 1

        a_i = P_original - P_i
        sum_attributions += a_i

        attr = FeatureAttribution(
            feature_name=fname,
            index=i,
            value=float(X[0, i]),
            baseline=float(training_medians[i]),
            attribution=round(a_i, 8),
            direction=_direction(a_i),
        )
        attributions.append(attr)

    # ── Step 6: Interaction gap ───────────────────────────────────────────
    total_shift = P_original - P_baseline
    sum_attributions = sum(f.attribution for f in attributions)
    interaction_gap = total_shift - sum_attributions

    # ── Step 7: Sort for top drivers ──────────────────────────────────────
    sorted_by_attr = sorted(attributions, key=lambda a: a.attribution, reverse=True)
    top_hallucination_drivers = [a for a in sorted_by_attr if a.attribution > 0][:top_k]
    top_protective_drivers = [a for a in reversed(sorted_by_attr) if a.attribution < 0][:top_k]

    decision_margin = P_original - threshold

    result = LocalAttributionResult(
        original_probability=P_original,
        baseline_probability=P_baseline,
        threshold=threshold,
        decision_margin=decision_margin,
        interaction_gap=interaction_gap,
        features=attributions,
        top_hallucination_drivers=top_hallucination_drivers,
        top_protective_drivers=top_protective_drivers,
        inference_count=inference_count,
    )

    logger.info(
        "local_attribution_computed",
        P_original=round(P_original, 4),
        P_baseline=round(P_baseline, 4),
        decision_margin=round(decision_margin, 4),
        interaction_gap=round(interaction_gap, 6),
        inference_count=inference_count,
        top_hallucination_driver=(
            top_hallucination_drivers[0].feature_name if top_hallucination_drivers else "none"
        ),
    )

    return result
