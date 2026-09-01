"""Single-Label Root Cause Classifier for HalluciSense Phase 25.

Assigns exactly one primary root-cause failure label to any non-verified or hallucinated prediction,
enabling automated taxonomy reporting and error analysis dashboards.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)


class RootCauseCategory(str, Enum):
    VERIFIED = "VERIFIED"
    RETRIEVAL_FAILURE = "Retrieval Failure"
    CLAIM_EXTRACTION_FAILURE = "Claim Extraction Failure"
    EVIDENCE_MISSING = "Evidence Missing"
    EVIDENCE_RANKING_FAILURE = "Evidence Ranking Failure"
    FACTUAL_CONTRADICTION = "Factual Contradiction"
    ENTITY_LINKING_FAILURE = "Entity Linking Failure"
    NLI_FAILURE = "NLI Failure"
    CONFIDENCE_FAILURE = "Confidence Failure"
    CONSISTENCY_FAILURE = "Consistency Failure"
    FUSION_FAILURE = "Fusion Failure"
    CALIBRATION_FAILURE = "Calibration Failure"
    KNOWLEDGE_GAP = "Knowledge Gap"
    UNSUPPORTED_DOMAIN = "Unsupported Domain"
    AMBIGUOUS_QUESTION = "Ambiguous Question"


class RootCauseClassifier:
    """Classifies pipeline predictions into a single, explainable root-cause failure label."""

    @classmethod
    def classify(
        cls,
        h_score: float,
        p1_res: Any,
        p2_res: Any,
        p3_res: Any,
        evidence_items: List[Any],
        query: str = "",
        response_text: str = "",
    ) -> RootCauseCategory:
        """Assign single-label failure classification based on empirical pipeline signals."""
        import time
        t_r0 = time.perf_counter()
        res = cls._classify_internal(h_score, p1_res, p2_res, p3_res, evidence_items, query, response_text)
        cls.last_risk_ms = round((time.perf_counter() - t_r0) * 1000.0, 2)
        return res

    @staticmethod
    def _classify_internal(
        h_score: float,
        p1_res: Any,
        p2_res: Any,
        p3_res: Any,
        evidence_items: List[Any],
        query: str = "",
        response_text: str = "",
    ) -> RootCauseCategory:

        # Verified threshold
        if h_score < 0.35:
            return RootCauseCategory.VERIFIED

        fe_score = float(getattr(p1_res, "factual_error_score", 0.5))
        evidence_count = len(evidence_items)

        # 1. Evidence Missing
        if evidence_count == 0:
            return RootCauseCategory.EVIDENCE_MISSING

        # 2. Claim Extraction Failure
        claims = getattr(p1_res, "claims", [])
        if not claims or any(len(c.split()) < 2 for c in claims):
            return RootCauseCategory.CLAIM_EXTRACTION_FAILURE

        # 3. Retrieval / Evidence Ranking Failure
        max_sim = max([getattr(e, "similarity_score", getattr(e, "score", 0.0)) for e in evidence_items], default=0.0)
        if max_sim < 0.35:
            return RootCauseCategory.RETRIEVAL_FAILURE
        elif max_sim < 0.60:
            return RootCauseCategory.EVIDENCE_RANKING_FAILURE

        # 4. Direct Factual Contradiction
        if fe_score >= 0.75:
            return RootCauseCategory.FACTUAL_CONTRADICTION

        # 5. NLI Neutral Ambiguity Failure
        if 0.45 <= fe_score <= 0.55:
            return RootCauseCategory.NLI_FAILURE

        # 6. Pillar 2 Confidence Failure
        if p2_res and getattr(p2_res, "available", False):
            cg_score = getattr(p2_res, "confidence_gap_score", 0.0) or 0.0
            if cg_score > 0.60:
                return RootCauseCategory.CONFIDENCE_FAILURE

        # 7. Pillar 3 Consistency Failure
        if p3_res and getattr(p3_res, "available", False):
            cf_score = getattr(p3_res, "consistency_failure_score", 0.0) or 0.0
            if cf_score > 0.60:
                return RootCauseCategory.CONSISTENCY_FAILURE

        # Default fallback to Knowledge Gap
        return RootCauseCategory.KNOWLEDGE_GAP
