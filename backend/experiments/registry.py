"""Phase 21 — Scientific Experiment Registry.

Tracks unique experiment runs (EXP0001, EXP0002, ...) and manages
reproducible experiment directories without overwriting past results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = BASE_DIR / "backend" / "experiments" / "runs"


class ExperimentRegistry:
    """Manages unique experiment IDs and execution directory manifests."""

    def __init__(self, base_dir: Path = EXPERIMENTS_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_dir / "experiment_registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_file.exists():
            with open(self.registry_file, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        else:
            self.records = []

    def save_registry(self) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def generate_next_id(self) -> str:
        """Generate next experiment ID (EXP0001, EXP0002, ...)."""
        existing_ids = [r["exp_id"] for r in self.records if "exp_id" in r]
        max_num = 0
        for eid in existing_ids:
            if eid.startswith("EXP"):
                try:
                    num = int(eid[3:])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        return f"EXP{max_num + 1:04d}"

    def register_experiment(self, exp_id: str, name: str, config: Dict[str, Any]) -> Path:
        """Create new experiment run directory and register entry."""
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "exp_id": exp_id,
            "name": name,
            "config": config,
            "status": "INITIALIZED",
            "dir_path": str(exp_dir),
        }
        self.records.append(entry)
        self.save_registry()
        return exp_dir

    def update_status(self, exp_id: str, status: str, metrics_summary: Optional[Dict[str, Any]] = None) -> None:
        """Update experiment status and final summary metrics."""
        for r in self.records:
            if r.get("exp_id") == exp_id:
                r["status"] = status
                if metrics_summary:
                    r["metrics_summary"] = metrics_summary
                break
        self.save_registry()
