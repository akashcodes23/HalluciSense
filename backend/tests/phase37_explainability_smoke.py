"""Phase 37 explainability regression tests.

These tests use deterministic stand-ins for the scaler/classifier so the
attribution algorithm itself can be validated without loading production ML
artifacts.
"""

import numpy as np
import pytest

from app.core.inference.explainability import (
    compute_local_feature_attributions,
    compute_shap_feature_attributions,
)


class FakeScaler:
    center_ = np.zeros(19, dtype=float)

    def transform(self, x):
        return np.asarray(x, dtype=float) - self.center_


class FakeClassifier:
    def predict_proba(self, x):
        # Deterministic nonlinear classifier. Feature 10 is risk-positive,
        # feature 11 is protective, and feature 14 creates an interaction.
        x = np.asarray(x, dtype=float)
        z = (
            2.0 * x[:, 10]
            - 1.5 * x[:, 11]
            + 0.5 * x[:, 14]
            + 0.25 * x[:, 10] * x[:, 14]
        )
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


def test_local_attribution_has_19_features_and_correct_directions():
    scaler = FakeScaler()
    clf = FakeClassifier()
    x = np.zeros(19, dtype=float)
    x[10] = 1.0
    x[11] = 0.5

    result = compute_local_feature_attributions(
        X_raw=x,
        scaler=scaler,
        clf=clf,
        threshold=0.54,
    )

    assert result["available"] is True
    assert result["feature_count"] == 19
    assert result["baseline_method"] == "training_median_from_RobustScaler_center"
    assert len(result["features"]) == 19
    assert result["observed_probability"] > result["baseline_probability"]

    by_name = {item["feature"]: item for item in result["features"]}
    assert by_name["prob_p1"]["delta"] > 0
    assert by_name["prob_p2"]["delta"] < 0
    assert by_name["prob_p1"]["direction"] == "increases_hallucination"
    assert by_name["prob_p2"]["direction"] == "decreases_hallucination"


def test_local_attribution_reports_nonlinear_interaction_gap():
    scaler = FakeScaler()
    clf = FakeClassifier()
    x = np.zeros(19, dtype=float)
    x[10] = 1.0
    x[14] = 1.0

    result = compute_local_feature_attributions(x, scaler, clf, threshold=0.54)

    # Independent leave-one-out deltas cannot exactly reconstruct a nonlinear
    # interaction. The residual must therefore be surfaced rather than hidden.
    assert abs(result["interaction_gap"]) > 1e-6
    assert "not SHAP" in result["non_additivity_note"]


def test_backward_compatible_feature_importance_uses_same_local_delta():
    scaler = FakeScaler()
    clf = FakeClassifier()
    x = np.zeros((1, 19), dtype=float)
    x[0, 10] = 1.0

    legacy = compute_shap_feature_attributions(x, scaler, clf)
    modern = compute_local_feature_attributions(x, scaler, clf)
    modern_map = {item["feature"]: round(item["delta"], 4) for item in modern["features"]}

    assert legacy == modern_map
    assert len(legacy) == 19


def test_wrong_feature_dimension_is_rejected():
    with pytest.raises(ValueError, match="Expected 19 features"):
        compute_local_feature_attributions(np.zeros(18), FakeScaler(), FakeClassifier())
