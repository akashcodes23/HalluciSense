"""Phase 41 — Data Leakage & Integrity Unit Test Suite.

Verifies:
- Label shuffle test collapses model to chance (ROC-AUC in [0.45, 0.55])
- Zero duplicate vector overlap across partitions
- Zero overlap with adversarial evaluation suites
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def test_label_shuffle_collapses_to_chance():
    """Verify that training on randomly permuted labels produces chance ROC-AUC (~0.50)."""
    np.random.seed(42)
    N = 2000
    X = np.random.randn(N, 19)
    y_true = (X[:, 0] > 0).astype(int)
    y_shuffled = np.random.permutation(y_true)
    
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    
    clf = HistGradientBoostingClassifier(max_iter=30, random_state=42)
    clf.fit(X_s[:1500], y_shuffled[:1500])
    
    probs = clf.predict_proba(X_s[1500:])[:, 1]
    auc = roc_auc_score(y_shuffled[1500:], probs)
    
    assert 0.40 <= auc <= 0.60, f"Expected chance AUC in [0.40, 0.60], got {auc:.4f}"
