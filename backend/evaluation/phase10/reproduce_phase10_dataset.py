"""Deterministic reproduction script for Phase 10 datasets."""

from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from evaluation.phase10.build_phase10_dataset import (
    build_750_scientific_dataset,
    build_250_adversarial_adaptive_dataset,
)

if __name__ == "__main__":
    records_750, report = build_750_scientific_dataset()
    records_250 = build_250_adversarial_adaptive_dataset()
    print("✓ Phase 10 Datasets successfully reproduced.")
