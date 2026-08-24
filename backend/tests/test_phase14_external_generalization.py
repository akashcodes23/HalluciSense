"""Phase 14 External Generalization & Zero-Tuning Validation Tests."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BACKEND_DIR / "evaluation" / "phase14" / "external_dataset_manifest.json"
FROZEN_CONFIG_PATH = BACKEND_DIR / "evaluation" / "phase14" / "phase14_external_frozen_config.json"


class TestPhase14ExternalGeneralization:
    def test_external_dataset_manifest_integrity(self):
        """Verifies external benchmark manifest lists all 5 required peer-reviewed datasets."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        dataset_names = [d["dataset_name"] for d in manifest["datasets"]]
        assert "TruthfulQA" in dataset_names
        assert "HaluEval" in dataset_names
        assert "FEVER" in dataset_names
        assert "RAGTruth" in dataset_names
        assert "BioASQ-FactCheck" in dataset_names

    def test_frozen_configuration_preserves_base_weights(self):
        """Verifies zero-tuning protocol: base weights remain fixed at alpha=0.40, beta=0.30, gamma=0.30."""
        with open(FROZEN_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        assert cfg["base_weights"]["alpha_factual_error"] == 0.40
        assert cfg["base_weights"]["beta_confidence_gap"] == 0.30
        assert cfg["base_weights"]["gamma_consistency_failure"] == 0.30
        assert cfg["calibration_parameters"]["method"] == "platt"
