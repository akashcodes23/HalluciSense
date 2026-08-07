"""NLI Diagnostics Engine for HalluciSense Phase 25 (Part 4).

Audits premise-hypothesis pair classifications across DeBERTa-v3-small.
Measures false entailments, false contradictions, and neutral ambiguity rates.
Generates nli_report.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import structlog

from app.core.engine.entailment import EvidenceEntailmentEngine

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results" / "phase25"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_nli_diagnostics(pairs: List[Dict[str, str]]) -> Dict[str, Any]:
    """Audit NLI model outputs across claim-evidence pairs."""
    logger.info("run_nli_diagnostics_start", num_pairs=len(pairs))

    nli_engine = EvidenceEntailmentEngine()

    entailment_probs = []
    neutral_probs = []
    contradiction_probs = []
    labels = []

    false_entailments = 0
    false_contradictions = 0
    neutral_ambiguities = 0

    for pair in pairs:
        claim = pair.get("claim", "")
        evidence = pair.get("evidence", "")
        expected = pair.get("expected_label", "entailment")

        res = nli_engine.classify(claim=claim, evidence=evidence)
        ent = res["entailment"]
        neu = res["neutral"]
        con = res["contradiction"]

        entailment_probs.append(ent)
        neutral_probs.append(neu)
        contradiction_probs.append(con)

        pred_label = "entailment" if ent >= max(neu, con) else ("contradiction" if con >= max(ent, neu) else "neutral")
        labels.append(pred_label)

        if pred_label == "entailment" and expected == "contradiction":
            false_entailments += 1
        elif pred_label == "contradiction" and expected == "entailment":
            false_contradictions += 1
        elif pred_label == "neutral" and neu > 0.70:
            neutral_ambiguities += 1

    total = max(1, len(pairs))
    metrics = {
        "mean_entailment_prob": round(float(np.mean(entailment_probs)), 4),
        "mean_neutral_prob": round(float(np.mean(neutral_probs)), 4),
        "mean_contradiction_prob": round(float(np.mean(contradiction_probs)), 4),
        "false_entailment_rate": round(false_entailments / float(total), 4),
        "false_contradiction_rate": round(false_contradictions / float(total), 4),
        "neutral_ambiguity_rate": round(neutral_ambiguities / float(total), 4),
        "total_pairs_evaluated": len(pairs),
    }

    # Save nli_report.md
    report_md = f"""# HalluciSense Natural Language Inference (NLI) Diagnostic Report

## Model Configuration
* **NLI Model**: `cross-encoder/nli-deberta-v3-small`
* **Evaluated Pairs**: `{len(pairs)}`

## Performance Metrics

| Diagnostic Metric | Value | Target | Status |
|:---|:---:|:---:|:---:|
| **Mean Entailment Probability** | `{metrics['mean_entailment_prob']:.4f}` | - | - |
| **Mean Neutral Probability** | `{metrics['mean_neutral_prob']:.4f}` | - | - |
| **Mean Contradiction Probability** | `{metrics['mean_contradiction_prob']:.4f}` | - | - |
| **False Entailment Rate** | `{metrics['false_entailment_rate']:.4f}` | $\\le 0.05$ | {"✅" if metrics['false_entailment_rate'] <= 0.05 else "⚠️"} |
| **False Contradiction Rate** | `{metrics['false_contradiction_rate']:.4f}` | $\\le 0.05$ | {"✅" if metrics['false_contradiction_rate'] <= 0.05 else "⚠️"} |
| **Neutral Ambiguity Rate** | `{metrics['neutral_ambiguity_rate']:.4f}` | $\\le 0.15$ | {"✅" if metrics['neutral_ambiguity_rate'] <= 0.15 else "⚠️"} |
"""
    with open(REPORTS_DIR / "nli_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    with open(RESULTS_DIR / "nli_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
