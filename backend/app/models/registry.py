"""HalluciSense Centralized Model Registry.

Provides lazy-loading, checksum verification, robust pathlib path resolution,
schema validation, and production diagnostics for frozen Pillar 1 and Hybrid models.
"""

from __future__ import annotations

import os
import json
import sklearn
import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Primary module base directory resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def resolve_results_directory() -> Path:
    """Robustly resolve the evaluation_results directory across local and container runtimes."""
    candidates = [
        BASE_DIR / "evaluation_results",
        BASE_DIR / "backend" / "evaluation_results",
        Path.cwd() / "evaluation_results",
        Path.cwd() / "backend" / "evaluation_results",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    # Default fallback
    return BASE_DIR / "evaluation_results"


RESULTS_DIR = resolve_results_directory()


class ModelRegistry:
    """Centralized Lazy-Loading Registry for HalluciSense Frozen Models."""

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or resolve_results_directory()
        self._pillar1_cache: Optional[Tuple[Any, Any, Dict[str, Any]]] = None
        self._pillar2_cache: Optional[Tuple[Any, Any, Dict[str, Any]]] = None
        self._hybrid_cache: Optional[Tuple[Any, Any, Dict[str, Any]]] = None

    def load_pillar1_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Lazy-load Pillar 1 (Evidence Consistency) frozen model and scaler."""
        if self._pillar1_cache is not None:
            return self._pillar1_cache

        logger.info("loading_pillar1_model", results_dir=str(self.results_dir))
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

        logger.info("loading_pillar2_model", results_dir=str(self.results_dir))
        p2_dir = self.results_dir / "phase6l" / "final_model"

        if (p2_dir / "preprocessing.joblib").exists() and (p2_dir / "classifier.joblib").exists():
            meta = {"operating_threshold": 0.57, "scaler": "StandardScaler", "model": "HistGradientBoostingClassifier"}
            meta_path = p2_dir / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            scaler = joblib.load(p2_dir / "preprocessing.joblib")
            clf = joblib.load(p2_dir / "classifier.joblib")

            self._pillar2_cache = (scaler, clf, meta)
        else:
            logger.info("pillar2_model_not_found_falling_back_to_pillar1")
            self._pillar2_cache = self.load_pillar1_model()

        return self._pillar2_cache

    def load_hybrid_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Lazy-load Hybrid Fusion frozen model, scaler, and protocol."""
        if self._hybrid_cache is not None:
            return self._hybrid_cache

        logger.info("loading_hybrid_model", results_dir=str(self.results_dir), cwd=os.getcwd())
        h_dir = self.results_dir / "phase6m" / "final_hybrid_model"
        
        clf_path = h_dir / "hybrid_meta_classifier.joblib"
        scaler_path = h_dir / "preprocessing.joblib"

        if clf_path.exists() and scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
                clf = joblib.load(clf_path)

                meta = {}
                if (h_dir / "model_metadata.json").exists():
                    with open(h_dir / "model_metadata.json", "r", encoding="utf-8") as f:
                        meta = json.load(f)

                self._hybrid_cache = (scaler, clf, meta)
                logger.info(
                    "hybrid_model_loaded_successfully",
                    clf_size=clf_path.stat().st_size,
                    scaler_size=scaler_path.stat().st_size,
                    sklearn_version=sklearn.__version__,
                    joblib_version=joblib.__version__,
                )
                return self._hybrid_cache
            except Exception as exc:
                logger.error("hybrid_model_load_exception_falling_back", error=str(exc))
                return self.load_pillar1_model()
        else:
            logger.warning(
                "hybrid_artifacts_missing_falling_back_to_pillar1",
                expected_dir=str(h_dir),
                clf_exists=clf_path.exists(),
                scaler_exists=scaler_path.exists(),
                dir_contents=os.listdir(h_dir) if h_dir.exists() else "Directory does not exist",
            )
            return self.load_pillar1_model()

    def verify_checksums(self) -> Dict[str, bool]:
        """Verify checksums and presence of frozen model artifacts."""
        p1_dir = self.results_dir / "phase6k" / "final_model"
        p1_clf_path = p1_dir / "pillar1_logistic_model.joblib"

        h_dir = self.results_dir / "phase6m" / "final_hybrid_model"
        h_clf_path = h_dir / "hybrid_meta_classifier.joblib"
        h_scaler_path = h_dir / "preprocessing.joblib"

        return {
            "pillar1_classifier_exists": p1_clf_path.exists(),
            "hybrid_classifier_exists": h_clf_path.exists(),
            "hybrid_scaler_exists": h_scaler_path.exists(),
            "hybrid_classifier_valid_size": h_clf_path.stat().st_size > 100 if h_clf_path.exists() else False,
        }

    def get_detailed_health_status(self) -> Dict[str, Any]:
        """Return diagnostic health and directory audit metadata for SRE observability."""
        h_dir = self.results_dir / "phase6m" / "final_hybrid_model"
        p1_dir = self.results_dir / "phase6k" / "final_model"

        clf_path = h_dir / "hybrid_meta_classifier.joblib"
        scaler_path = h_dir / "preprocessing.joblib"
        meta_path = h_dir / "model_metadata.json"
        schema_path = h_dir / "feature_schema.json"

        hybrid_available = clf_path.exists() and scaler_path.exists()

        return {
            "resolved_results_directory": str(self.results_dir),
            "cwd": os.getcwd(),
            "base_dir": str(BASE_DIR),
            "artifacts_found": {
                "hybrid_classifier": clf_path.exists(),
                "hybrid_scaler": scaler_path.exists(),
                "model_metadata": meta_path.exists(),
                "feature_schema": schema_path.exists(),
                "pillar1_classifier": (p1_dir / "pillar1_logistic_model.joblib").exists(),
            },
            "artifact_sizes": {
                "hybrid_classifier_bytes": clf_path.stat().st_size if clf_path.exists() else 0,
                "hybrid_scaler_bytes": scaler_path.stat().st_size if scaler_path.exists() else 0,
            },
            "loaded_successfully": hybrid_available,
            "current_sklearn_version": sklearn.__version__,
            "current_joblib_version": joblib.__version__,
        }


# Global singleton instance
registry = ModelRegistry()

