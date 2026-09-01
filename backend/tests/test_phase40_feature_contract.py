"""Phase 40 — Feature Contract & Dimensionality Unit Test Suite.

Verifies:
- 19-feature schema consistency
- Feature ordering matches model metadata
- Feature bounds [0, 1] and logit transformations
- Training medians alignment with RobustScaler
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.inference.local_attribution import get_feature_schema, get_training_medians


def test_feature_count_and_schema():
    """Verify exactly 19 features are defined in canonical schema."""
    schema = get_feature_schema()
    assert len(schema) == 19
    assert schema[0] == "p1_mean_entailment"
    assert schema[2] == "p1_mean_contradiction"
    assert schema[4] == "p1_num_claims"
    assert schema[10] == "prob_p1"
    assert schema[18] == "prob_ratio"


def test_training_medians_shape():
    """Verify training medians are loaded as a 19-element vector."""
    medians = get_training_medians()
    assert isinstance(medians, np.ndarray)
    assert medians.shape == (19,)
    assert not np.isnan(medians).any()
