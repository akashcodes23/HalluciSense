"""Multi-Venue Submission Packager for HalluciSense Phase 27 (Part 15).

Builds camera-ready submission packages for:
1. Elsevier (Information Fusion / Artificial Intelligence)
2. Springer (Knowledge-Based Systems / Neural Processing)
3. IEEE (Transactions on AI)
4. NeurIPS (Benchmark & Artifact Track)
5. ACL / EMNLP (Main Conference & Findings)
6. ICML
7. ICLR

Outputs zip archives to submission_packages/
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Any

import structlog

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = BASE_DIR / "submission_packages"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

VENUES = [
    "elsevier",
    "springer",
    "ieee",
    "neurips",
    "acl",
    "icml",
    "iclr",
]

CORE_FILES = [
    "reproducibility_audit.md",
    "final_scientific_audit.md",
    "dataset_provenance.json",
    "model_provenance.json",
    "environment_snapshot.json",
    "CITATION.cff",
    "codemeta.json",
    "LICENSE",
    "README.md",
]


def build_venue_package(venue_name: str) -> Path:
    """Build standardized venue submission zip package."""
    zip_path = SUBMISSION_DIR / f"{venue_name}_submission.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Core audit & provenance metadata
        for fname in CORE_FILES:
            fpath = BASE_DIR / fname
            if fpath.exists():
                zf.write(fpath, arcname=f"metadata/{fname}")

        # 2. Add reports
        reports_dir = BASE_DIR / "reports"
        if reports_dir.exists():
            for root, _, files in os.walk(reports_dir):
                for file in files:
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(BASE_DIR)
                    zf.write(full_p, arcname=str(rel_p))

        # 3. Add evaluation results & LaTeX tables
        eval_dir = BASE_DIR / "backend" / "evaluation_results"
        if eval_dir.exists():
            for root, _, files in os.walk(eval_dir):
                for file in files:
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(BASE_DIR)
                    zf.write(full_p, arcname=str(rel_p))

        # 4. Add artifact evaluation guide
        art_dir = BASE_DIR / "artifact"
        if art_dir.exists():
            for root, _, files in os.walk(art_dir):
                for file in files:
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(BASE_DIR)
                    zf.write(full_p, arcname=str(rel_p))

    logger.info("venue_package_built", venue=venue_name, zip_path=str(zip_path))
    return zip_path


def build_all_submission_packages() -> List[str]:
    """Build submission packages for all target venues."""
    built_packages = []
    print("=" * 80)
    print("HALLUCISENSE MULTI-VENUE SUBMISSION PACKAGER")
    print("=" * 80)

    for venue in VENUES:
        zpath = build_venue_package(venue)
        built_packages.append(str(zpath))
        print(f"  - [{venue.upper()}] Package Built: {zpath.name} ({zpath.stat().st_size / 1024:.1f} KB)")

    print("=" * 80)
    print(f"✅ SUCCESSFULLY BUILT {len(built_packages)} VENUE SUBMISSION PACKAGES IN submission_packages/")
    print("=" * 80)
    return built_packages


if __name__ == "__main__":
    build_all_submission_packages()
