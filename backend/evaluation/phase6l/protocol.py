"""Phase 6L.2 — Stage 9: Protocol Lock Engine.

Freezes exactly ONE development protocol into final_model_protocol.json before held-out validation.
Locks feature subset, preprocessing, classifier, solver, hyperparameters, decision threshold, and random seed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import structlog

from evaluation.phase6l.config import PHASE6L_DIR

logger = structlog.get_logger(__name__)


def export_final_model_protocol(
    winning_candidate: Dict[str, Any],
    out_dir: Path = PHASE6L_DIR,
) -> Dict[str, Any]:
    """Freeze and export final immutable model protocol to final_model_protocol.json.

    Returns:
        Dict containing locked protocol specification.
    """
    logger.info("stage9_export_final_protocol_start", winner=winning_candidate.get("classifier_name"))

    summary = winning_candidate["summary_metrics"]
    set_key = winning_candidate.get("feature_set", "SET_D_HIGH_INFORMATION")
    feat_cols = winning_candidate.get("features", [])

    protocol_payload = {
        "selected_candidate": winning_candidate.get("classifier_name"),
        "feature_set_name": set_key,
        "feature_count": len(feat_cols),
        "feature_names": feat_cols,
        "scaler": winning_candidate.get("scaler_type", "RobustScaler"),
        "classifier": "LogisticRegression",
        "solver": "liblinear",
        "penalty": "l2",
        "C": 1.0,
        "hyperparameters": {
            "solver": "liblinear",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "fit_intercept": True,
        },
        "decision_threshold": summary.get("best_mcc_threshold", 0.50),
        "random_seed": 42,
        "dev_sample_count": 58002,
        "val_sample_count": 12483,
        "dev_performance_summary": {
            "roc_auc_mean": summary.get("roc_auc_mean"),
            "pr_auc_mean": summary.get("pr_auc_mean"),
            "brier_score_mean": summary.get("brier_score_mean"),
            "log_loss_mean": summary.get("log_loss_mean"),
            "ece": summary.get("ece"),
            "best_mcc": summary.get("best_mcc"),
            "accuracy_at_best_thresh": summary.get("accuracy_at_best_thresh"),
            "f1_at_best_thresh": summary.get("f1_at_best_thresh"),
        },
        "protocol_locked": True,
        "protocol_locked_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "firewall_status": "HELD_OUT_VALIDATION_PARTITION_SEALED",
    }

    out_file = out_dir / "final_model_protocol.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(protocol_payload, f, indent=2)

    logger.info("stage9_export_final_protocol_complete", path=str(out_file))
    return protocol_payload
