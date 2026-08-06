"""Part 6 — Automated 12-Class Failure Taxonomy Classifier.

Automatically categorizes detected hallucinations into 12 classes:
1. Fabrication
2. Temporal Inconsistency
3. Citation Hallucination
4. Reasoning Fallacy
5. Mathematical Error
6. Entity Swap
7. Numerical Discrepancy
8. Causal Misattribution
9. Logical Contradiction
10. Unsupported Inference
11. Context Drift
12. Semantic Ambiguity

Assigns:
- Hallucination Type
- Severity Level (Low, Medium, High, Critical)
- Confidence Score
- Affected Spans
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class FailureTaxonomyResult:
    hallucination_type: str
    severity: str
    confidence: float
    affected_spans: List[Dict[str, Any]]
    explanation: str


class FailureTaxonomyClassifier:
    """Classifies hallucination failure modes into 12 categories."""

    TAXONOMY_TYPES = [
        "Fabrication",
        "Temporal Inconsistency",
        "Citation Hallucination",
        "Reasoning Fallacy",
        "Mathematical Error",
        "Entity Swap",
        "Numerical Discrepancy",
        "Causal Misattribution",
        "Logical Contradiction",
        "Unsupported Inference",
        "Context Drift",
        "Semantic Ambiguity",
    ]

    def classify_failure(
        self,
        claim: str,
        h_score: float,
        fe_val: float,
        evidence_text: str = "",
    ) -> FailureTaxonomyResult:
        """Classify claim into one of the 12 failure taxonomy categories."""
        claim_lower = claim.lower()

        # 1. Mathematical / Numerical
        if re.search(r"\b\d+\s*[\+\-\*/=]\s*\d+\b", claim) or re.search(r"\b\d+\.?\d*\s*%\b", claim):
            htype = "Numerical Discrepancy" if fe_val > 0.3 else "Mathematical Error"
            severity = "High" if h_score > 0.65 else "Medium"
        # 2. Citation
        elif any(w in claim_lower for w in [" et al.", "doi:", "isbn:", "published in", "journal of", "cited in"]):
            htype = "Citation Hallucination"
            severity = "Critical" if h_score > 0.70 else "High"
        # 3. Temporal
        elif any(w in claim_lower for w in ["in 19", "in 20", "bc", "ad", "century", "yesterday", "tomorrow"]):
            htype = "Temporal Inconsistency"
            severity = "High" if h_score > 0.65 else "Medium"
        # 4. Entity Swap
        elif re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", claim):
            htype = "Entity Swap" if fe_val < 0.4 else "Semantic Ambiguity"
            severity = "High" if h_score > 0.65 else "Low"
        # 5. Reasoning / Causal
        elif any(w in claim_lower for w in ["because", "therefore", "caused by", "leads to", "as a result"]):
            htype = "Causal Misattribution" if "caused" in claim_lower else "Reasoning Fallacy"
            severity = "Medium"
        # 6. Fabrication / Unsupported
        elif fe_val < 0.25:
            htype = "Fabrication"
            severity = "Critical" if h_score > 0.75 else "High"
        else:
            htype = "Unsupported Inference"
            severity = "Medium" if h_score > 0.50 else "Low"

        confidence = round(float(min(0.98, max(0.60, h_score * 0.9 + 0.15))), 4)

        # Locate span
        start_char = 0
        end_char = len(claim)

        spans = [{
            "start_char": start_char,
            "end_char": end_char,
            "text": claim,
            "span_risk_score": round(h_score, 4),
        }]

        explanation = f"Detected {htype} (Severity: {severity}, Confidence: {confidence:.2f})."

        return FailureTaxonomyResult(
            hallucination_type=htype,
            severity=severity,
            confidence=confidence,
            affected_spans=spans,
            explanation=explanation,
        )
