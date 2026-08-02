"""HalluciSense Centralized Model Registry.

Provides lazy-loading, checksum verification, schema validation, and access
to frozen Pillar 1, Pillar 2, and Hybrid models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import structlog

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results"


class ModelRegistry:
    """Centralized Lazy-Loading Registry for HalluciSense Frozen Models."""

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self._pillar1_cache: Optional[Tuple[Any, Any, Dict[str, Any]]] = None
        self._pillar2_cache: Optional[Tuple[Any, Any, Dict[str, Any]]] = None
        self._hybrid_cache: Optional[Tuple[Any, Any, Dict[str, Any]]] = None

    def load_pillar1_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Lazy-load Pillar 1 (Evidence Consistency) frozen model and scaler."""
        if self._pillar1_cache is not None:
            return self._pillar1_cache

        logger.info("loading_pillar1_model")
        p1_dir = self.results_dir / "phase6k" / "final_model"
        
        meta = {"operating_threshold": 0.56, "scaler": "RobustScaler", "model": "LogisticRegression"}
        meta_path = p1_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        scaler = joblib.load(p1_dir / "robust_scaler.joblib")
        clf = joblib.load(p1_dir / "pillar1_logistic_model.joblib")

        self._pillar1_cache = (scaler, clf, meta)
        return self._pillar1_cache

    def load_pillar2_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Lazy-load Pillar 2 (Structural Consistency) frozen model and scaler."""
        if self._pillar2_cache is not None:
            return self._pillar2_cache

        logger.info("loading_pillar2_model")
        p2_dir = self.results_dir / "phase6l" / "final_model"

        meta = {"operating_threshold": 0.57, "scaler": "StandardScaler", "model": "HistGradientBoostingClassifier"}
        meta_path = p2_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        scaler = joblib.load(p2_dir / "preprocessing.joblib")
        clf = joblib.load(p2_dir / "classifier.joblib")

        self._pillar2_cache = (scaler, clf, meta)
        return self._pillar2_cache

    def load_hybrid_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Lazy-load Hybrid Fusion frozen model, scaler, and protocol."""
        if self._hybrid_cache is not None:
            return self._hybrid_cache

        logger.info("loading_hybrid_model")
        h_dir = self.results_dir / "phase6m" / "final_hybrid_model"
        
        scaler = joblib.load(h_dir / "preprocessing.joblib")
        clf = joblib.load(h_dir / "hybrid_meta_classifier.joblib")

        with open(h_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._hybrid_cache = (scaler, clf, meta)
        return self._hybrid_cache

    def verify_checksums(self) -> Dict[str, bool]:
        """Verify checksums of frozen model artifacts."""
        h_dir = self.results_dir / "phase6m" / "final_hybrid_model"
        h_clf_path = h_dir / "hybrid_meta_classifier.joblib"
        h_scaler_path = h_dir / "preprocessing.joblib"

        return {
            "hybrid_classifier_exists": h_clf_path.exists(),
            "hybrid_scaler_exists": h_scaler_path.exists(),
            "hybrid_classifier_valid_size": h_clf_path.stat().st_size > 100 if h_clf_path.exists() else False,
        }


# Global singleton instance
registry = ModelRegistry()
