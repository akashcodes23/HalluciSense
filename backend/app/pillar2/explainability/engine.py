"""
HalluciSense Pillar 2 — Explainability Engine
=============================================
Generates comprehensive, structured, human-readable verification reports and recommendations.
"""

from typing import Dict, List

import structlog
from app.pillar2.claim_extraction.schemas import ExtractedClaim
from app.pillar2.consensus_engine.schemas import ConsensusResult
from app.pillar2.contradiction_analysis.schemas import ContradictionAnalysisResult
from app.pillar2.evidence_retrieval.schemas import EvidenceItem
from app.pillar2.explainability.schemas import ClaimAnalysisItem, SourceSummary, VerificationExplanation
from app.pillar2.feature_generation.schemas import PillarTwoFeatures
from app.pillar2.multi_llm_verifier.schemas import VerificationLabel
from app.pillar2.unified_hscore.schemas import RiskCategory, UnifiedHScoreResult

logger = structlog.get_logger(__name__)


class PillarTwoExplainabilityEngine:
    """
    Generates human-readable verification reports for human auditors and automated APIs.
    """

    def generate_explanation(
        self,
        claims: List[ExtractedClaim],
        evidence_items: List[EvidenceItem],
        consensus_map: Dict[str, ConsensusResult],
        contradiction_result: ContradictionAnalysisResult,
        p2_features: PillarTwoFeatures,
        hscore_result: UnifiedHScoreResult,
    ) -> VerificationExplanation:
        """
        Generate full VerificationExplanation.

        Returns
        -------
        VerificationExplanation
        """
        n_claims = len(claims)
        n_ev = len(evidence_items)

        # 1. Executive Summary
        exec_summary = (
            f"HalluciSense Verification Report: Analyzed {n_claims} atomic claim(s) against "
            f"{n_ev} retrieved evidence source(s). Assigned HalluciSense Score is "
            f"{hscore_result.hallucisense_score:.1f}/100 ({hscore_result.risk_category.value} Risk) "
            f"with {hscore_result.overall_confidence*100:.1f}% confidence. "
            f"Pillar 1 statistical probability: {hscore_result.pillar1_probability:.3f}. "
            f"Support ratio: {p2_features.support_ratio*100:.1f}%, Contradiction ratio: {p2_features.contradiction_ratio*100:.1f}%."
        )

        # 2. Claim Analysis Items
        claim_analysis_items: List[ClaimAnalysisItem] = []
        for claim in claims:
            cid = claim.claim_id
            consensus = consensus_map.get(cid)

            lbl_str = consensus.majority_label.value if consensus else "UNKNOWN"
            conf = consensus.consensus_confidence if consensus else 0.50

            sup_cnt = 0
            con_cnt = 0
            if consensus:
                for v_lbl in consensus.label_distribution.keys():
                    if v_lbl == VerificationLabel.SUPPORTED.value:
                        sup_cnt += consensus.label_distribution[v_lbl]
                    elif v_lbl == VerificationLabel.CONTRADICTED.value:
                        con_cnt += consensus.label_distribution[v_lbl]

            item_summary = (
                f"Claim '{claim.claim_text[:40]}...' verified as {lbl_str} "
                f"(confidence: {conf*100:.1f}%, supported by {sup_cnt} verifiers, "
                f"contradicted by {con_cnt} verifiers)."
            )

            claim_analysis_items.append(
                ClaimAnalysisItem(
                    claim_id=cid,
                    claim_text=claim.claim_text,
                    consensus_label=lbl_str,
                    confidence=conf,
                    supporting_count=sup_cnt,
                    contradicting_count=con_cnt,
                    summary=item_summary,
                )
            )

        # 3. Evidence Analysis
        ev_analysis = (
            f"Retrieved {n_ev} evidence items from {len(set(e.source for e in evidence_items))} provider(s). "
            f"Mean Authority Score: {p2_features.authority_score:.2f}, Citation Quality: {p2_features.citation_quality*100:.1f}%, "
            f"Source Diversity: {p2_features.source_diversity:.2f}, Evidence Coverage: {p2_features.evidence_coverage*100:.1f}%."
        )

        # 4. Supporting & Contradicting Sources
        supporting_sources: List[SourceSummary] = []
        contradicting_sources: List[SourceSummary] = []

        for e in evidence_items[:5]:
            src_obj = SourceSummary(
                title=e.title,
                url=e.url,
                provider=e.source,
                authority_score=e.authority_score,
            )
            supporting_sources.append(src_obj)

        if contradiction_result.contradictions:
            for cnt in contradiction_result.contradictions[:3]:
                if evidence_items:
                    e = evidence_items[0]
                    contradicting_sources.append(
                        SourceSummary(
                            title=f"Contradiction regarding '{cnt.claim_text[:30]}'",
                            url=e.url,
                            provider=e.source,
                            authority_score=e.authority_score,
                        )
                    )

        # 5. Actionable Recommendations
        recs: List[str] = []
        if hscore_result.risk_category in [RiskCategory.CRITICAL, RiskCategory.HIGH]:
            recs.append("REJECT LLM response: High probability of severe factual hallucination or contradiction.")
            recs.append("Flag response for human expert review before downstream ingestion.")
        elif hscore_result.risk_category == RiskCategory.MODERATE:
            recs.append("PROCEED WITH CAUTION: Response contains unverified or partially supported claims.")
            recs.append("Display inline warnings for claims with low consensus confidence.")
        else:
            recs.append("ACCEPT LLM response: High empirical grounding across multi-LLM verifiers and authoritative sources.")

        if p2_features.evidence_coverage < 0.8:
            recs.append("Expand retrieval query scope to cover unsupported atomic claims.")

        logger.info(
            "explanation_generated",
            risk_category=hscore_result.risk_category.value,
            hscore=hscore_result.hallucisense_score,
            recs_count=len(recs),
        )

        return VerificationExplanation(
            executive_summary=exec_summary,
            claim_analysis=claim_analysis_items,
            evidence_analysis=ev_analysis,
            supporting_sources=supporting_sources,
            contradicting_sources=contradicting_sources,
            confidence_score=hscore_result.overall_confidence,
            risk_category=hscore_result.risk_category,
            risk_score=hscore_result.hallucisense_score,
            actionable_recommendations=recs,
        )
