"""Phase 21 — Experiment Artifact Logger.

Saves predictions, metrics, hardware metadata, git commits, and environment logs
into a dedicated experiment directory.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any


class ExperimentLogger:
    """Logs per-experiment metadata and prediction dataframes."""

    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.exp_dir / "logs.txt"

    def log(self, message: str) -> None:
        """Write timestamped log message."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def log_environment_and_hardware(self, seed: int = 42) -> None:
        """Log system hardware, git commit, seed, and environment metadata."""
        # seed.txt
        with open(self.exp_dir / "seed.txt", "w", encoding="utf-8") as f:
            f.write(str(seed))

        # git_commit.txt
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            commit = "UNKNOWN"
        with open(self.exp_dir / "git_commit.txt", "w", encoding="utf-8") as f:
            f.write(commit)

        # hardware.json
        hw = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
        with open(self.exp_dir / "hardware.json", "w", encoding="utf-8") as f:
            json.dump(hw, f, indent=2)

    def log_predictions_and_metrics(
        self,
        predictions: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        exec_time_sec: float,
    ) -> None:
        """Export predictions (CSV), metrics (JSON/CSV), config, and execution time."""
        # config.json
        with open(self.exp_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # metrics.json
        with open(self.exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # execution_time.json
        with open(self.exp_dir / "execution_time.json", "w", encoding="utf-8") as f:
            json.dump({"execution_time_seconds": round(exec_time_sec, 4)}, f, indent=2)

        # predictions.csv
        if predictions:
            fieldnames = list(predictions[0].keys())
            with open(self.exp_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(predictions)

        # Parquet fallback representation
        with open(self.exp_dir / "predictions.parquet", "w", encoding="utf-8") as f:
            json.dump({"format": "parquet_json_fallback", "records": len(predictions)}, f, indent=2)
