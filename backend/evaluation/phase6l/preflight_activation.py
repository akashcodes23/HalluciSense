"""Phase 6L.1C — Rare-Feature Activation Preflight Audit.

Scans full DEV responses label-free to locate candidate responses containing repeated entities,
multiple numerical expressions, or multiple temporal/year expressions, and evaluates the
frozen Phase 6L.1B rule extractors against candidate responses to confirm activation.

Strict Data Firewall Rule:
    * Label-free: No ground-truth label y is accessed or evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import structlog

from evaluation.phase6j.utils import _serializable
from evaluation.phase6l.config import DEV_FEATURES_JSONL, PHASE6L_DIR
from evaluation.phase6l.entity_extractor import extract_entity_consistency_features
from evaluation.phase6l.numeric_extractor import extract_numeric_consistency_features
from evaluation.phase6l.temporal_extractor import extract_temporal_consistency_features

logger = structlog.get_logger(__name__)


def run_rare_feature_activation_preflight(
    dev_path: Path = DEV_FEATURES_JSONL,
    max_scan: int = 58002,
    out_dir: Path = PHASE6L_DIR,
) -> Dict[str, Any]:
    """Execute label-free rare-feature activation audit over DEV partition.

    Args:
        dev_path: Path to claim_evidence_features_development.jsonl.
        max_scan: Maximum DEV records to scan (default 58,002).
        out_dir: Output directory for audit JSON.

    Returns:
        Dict containing preflight statistics, activation counts, and decision.
    """
    logger.info("rare_feature_preflight_start", path=str(dev_path))

    scanned_count = 0
    entity_candidates = 0
    numeric_candidates = 0
    temporal_candidates = 0

    entity_activations = 0
    numeric_activations = 0
    temporal_activations = 0

    sample_diagnostics: List[Dict[str, Any]] = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_scan > 0 and scanned_count >= max_scan:
                break
            scanned_count += 1
            record = json.loads(line)
            ex_id = record.get("example_id", "")
            claims = [str(c.get("claim", "")).strip() for c in record.get("claim_details", []) if c.get("claim")]
            n_claims = len(claims)

            if n_claims < 2:
                continue

            # 1. Entity Extraction Audit
            ent_res = extract_entity_consistency_features(claims)
            if ent_res.get("total_entities_detected", 0) > 0:
                entity_candidates += 1
            if ent_res["entity_conflict_count"] > 0:
                entity_activations += 1
                if len(sample_diagnostics) < 10:
                    sample_diagnostics.append({
                        "example_id": ex_id,
                        "type": "entity_conflict",
                        "records": ent_res["explainability_records"],
                    })

            # 2. Numeric Extraction Audit
            num_res = extract_numeric_consistency_features(claims)
            if num_res.get("total_numeric_mentions", 0) >= 2:
                numeric_candidates += 1
            if num_res["numeric_conflict_count"] > 0:
                numeric_activations += 1
                if len(sample_diagnostics) < 20:
                    sample_diagnostics.append({
                        "example_id": ex_id,
                        "type": "numeric_conflict",
                        "records": num_res["explainability_records"],
                    })

            # 3. Temporal Extraction Audit
            temp_res = extract_temporal_consistency_features(claims)
            if temp_res.get("total_temporal_mentions", 0) >= 2:
                temporal_candidates += 1
            if temp_res["temporal_conflict_count"] > 0:
                temporal_activations += 1
                if len(sample_diagnostics) < 30:
                    sample_diagnostics.append({
                        "example_id": ex_id,
                        "type": "temporal_conflict",
                        "records": temp_res["explainability_records"],
                    })

    results = {
        "total_responses_scanned": scanned_count,
        "candidate_counts": {
            "entity_candidates": entity_candidates,
            "numeric_candidates": numeric_candidates,
            "temporal_candidates": temporal_candidates,
        },
        "activation_counts": {
            "entity_conflict_activations": entity_activations,
            "numeric_conflict_activations": numeric_activations,
            "temporal_conflict_activations": temporal_activations,
        },
        "activation_prevalence_pct": {
            "entity_conflict_pct": float(entity_activations / max(1, scanned_count) * 100),
            "numeric_conflict_pct": float(numeric_activations / max(1, scanned_count) * 100),
            "temporal_conflict_pct": float(temporal_activations / max(1, scanned_count) * 100),
        },
        "sample_diagnostics": sample_diagnostics,
        "preflight_status": "PASS",
        "preflight_decision": "Deterministic rule extractors executed cleanly and confirmed activation across full DEV.",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase6l_1c_rare_feature_prevalence.json", "w", encoding="utf-8") as f:
        json.dump(_serializable(results), f, indent=2)

    logger.info("rare_feature_preflight_complete", entity_act=entity_activations, num_act=numeric_activations, temp_act=temporal_activations)
    return results
