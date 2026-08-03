"""
HalluciSense Pillar 2 — Unified H-Score Calculator
===================================================
Fuses frozen Pillar 1 probability with Pillar 2 multi-LLM consensus and evidence features.
Does NOT modify or retrain Pillar 1.
"""

from typing import Dict, Optional

import structlog
from app.pillar2.contradiction_analysis.schemas import ContradictionAnalysisResult
from app.pillar2.feature_generation.schemas import PillarTwoFeatures
from app.pillar2.unified_hscore.schemas import RiskCategory, UnifiedHScoreResult

logger = structlog.get_logger(__name__)


class UnifiedHScoreCalculator:
    """
    Calculates the Unified HalluciSense Score (0-100).
    Fuses frozen Pillar 1 statistical probability with Pillar 2 evidence signals.
    """

    DEFAULT_WEIGHTS = {
        "pillar1_weight": 0.40,
        "contradiction_weight": 0.30,
        "evidence_quality_weight": 0.15,
        "consensus_weight": 0.15,
    }

    def calculate_hscore(
        self,
        pillar1_probability: float,
        p2_features: PillarTwoFeatures,
        contradiction_result: ContradictionAnalysisResult,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> UnifiedHScoreResult:
        """
        Compute 0-100 Unified HalluciSense Score.

        Parameters
        ----------
        pillar1_probability : float (0.0 to 1.0)
        p2_features : PillarTwoFeatures
        contradiction_result : ContradictionAnalysisResult
        custom_weights : Optional[Dict[str, float]]

        Returns
        -------
        UnifiedHScoreResult
        """
        weights = custom_weights or self.DEFAULT_WEIGHTS
        w_p1 = weights.get("pillar1_weight", 0.40)
        w_cnt = weights.get("contradiction_weight", 0.30)
        w_ev = weights.get("evidence_quality_weight", 0.15)
        w_cs = weights.get("consensus_weight", 0.15)

        p1_prob = max(0.0, min(1.0, pillar1_probability))

        # Pillar 2 Contradiction Score (0-100)
        # Contradiction ratio + max severity + fabrication index
        cnt_risk_ratio = (
            p2_features.contradiction_ratio * 0.4
            + contradiction_result.max_severity * 0.4
            + contradiction_result.fabrication_index * 0.2
        )
        contradiction_score = min(100.0, cnt_risk_ratio * 100.0)

        # Pillar 2 Evidence Risk Score (0-100)
        # Low support, low authority, low citation quality => high risk
        ev_quality = (
            p2_features.support_ratio * 0.4
            + p2_features.authority_score * 0.3
            + p2_features.citation_quality * 0.2
            + p2_features.source_diversity * 0.1
        )
        evidence_risk_score = (1.0 - ev_quality) * 100.0

        # Pillar 2 Consensus Risk Score (0-100)
        # Low consensus confidence or high contradiction ratio => high risk
        consensus_risk_score = (1.0 - p2_features.consensus_confidence) * 50.0 + p2_features.contradiction_ratio * 50.0

        # Fused Raw H-Score (0-100)
        p1_score = p1_prob * 100.0
        fused_hscore = (
            p1_score * w_p1
            + contradiction_score * w_cnt
            + evidence_risk_score * w_ev
            + consensus_risk_score * w_cs
        )
        final_hscore = round(max(0.0, min(100.0, fused_hscore)), 2)

        # Categorize Risk
        if final_hscore >= 80.0:
            risk_category = RiskCategory.CRITICAL
        elif final_hscore >= 60.0:
            risk_category = RiskCategory.HIGH
        elif final_hscore >= 40.0:
            risk_category = RiskCategory.MODERATE
        elif final_hscore >= 20.0:
            risk_category = RiskCategory.LOW
        else:
            risk_category = RiskCategory.VERY_LOW

        # Overall verification confidence
        confidence = round(
            p2_features.consensus_confidence * 0.5
            + p2_features.verification_completeness * 0.3
            + (1.0 - abs(p1_prob - 0.5) * 0.4) * 0.2,
            4,
        )

        explanation = (
            f"HalluciSense Unified H-Score: {final_hscore}/100 ({risk_category.value} Risk). "
            f"Pillar 1 Statistical Probability: {p1_prob:.3f}. "
            f"Pillar 2 Contradiction Severity: {contradiction_result.max_severity:.2f}. "
            f"Evidence Support Ratio: {p2_features.support_ratio*100:.1f}%."
        )

        logger.info(
            "unified_hscore_calculated",
            hscore=final_hscore,
            risk=risk_category.value,
            pillar1_prob=p1_prob,
            confidence=confidence,
        )

        return UnifiedHScoreResult(
            hallucisense_score=final_hscore,
            risk_category=risk_category,
            overall_confidence=confidence,
            pillar1_probability=p1_prob,
            evidence_score=round(evidence_risk_score, 2),
            consensus_score=round(consensus_risk_score, 2),
            contradiction_score=round(contradiction_score, 2),
            component_weights=weights,
            explanation_summary=explanation,
        )
