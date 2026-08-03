"""
HalluciSense Pillar 2 — Consensus Engine
========================================
Computes statistical consensus metrics across multi-LLM verification outputs.
Includes majority voting, weighted voting, agreement matrices, Shannon entropy,
confidence variance, and disagreement analysis.
"""

import math
from typing import Dict, List

import structlog
from app.pillar2.consensus_engine.schemas import ConsensusResult, DisagreementDetail
from app.pillar2.multi_llm_verifier.schemas import SingleClaimVerification, VerificationLabel

logger = structlog.get_logger(__name__)


class ConsensusEngine:
    """
    Statistical Consensus Engine for Multi-LLM verifications.
    """

    def compute_consensus(
        self, claim_id: str, verifications: List[SingleClaimVerification]
    ) -> ConsensusResult:
        """
        Compute consensus across verifier results for a single claim.

        Parameters
        ----------
        claim_id : str
        verifications : List[SingleClaimVerification]

        Returns
        -------
        ConsensusResult
        """
        if not verifications:
            return ConsensusResult(
                claim_id=claim_id,
                majority_label=VerificationLabel.UNKNOWN,
                weighted_label=VerificationLabel.UNKNOWN,
                consensus_confidence=0.50,
                label_distribution={},
                label_weights={},
                pairwise_agreement_score=0.0,
                shannon_entropy=0.0,
                confidence_variance=0.0,
                agreement_matrix={},
                disagreeing_verifiers=[],
                verdict_summary="No verifications available to compute consensus.",
            )

        # 1. Label counts and confidence weights
        label_dist: Dict[str, int] = {}
        label_weights: Dict[str, float] = {}
        confidences: List[float] = []

        for v in verifications:
            lbl_str = v.label.value
            label_dist[lbl_str] = label_dist.get(lbl_str, 0) + 1
            label_weights[lbl_str] = round(label_weights.get(lbl_str, 0.0) + v.confidence, 4)
            confidences.append(v.confidence)

        # Majority label
        majority_lbl_str = max(label_dist, key=lambda k: (label_dist[k], label_weights[k]))
        majority_label = VerificationLabel(majority_lbl_str)

        # Weighted label
        weighted_lbl_str = max(label_weights, key=lambda k: label_weights[k])
        weighted_label = VerificationLabel(weighted_lbl_str)

        # 2. Consensus confidence (weighted mean)
        total_weight = sum(label_weights.values())
        maj_weight = label_weights[majority_lbl_str]
        consensus_confidence = round(maj_weight / total_weight, 4) if total_weight > 0 else 0.50

        # 3. Shannon Entropy
        n_total = len(verifications)
        entropy = 0.0
        for count in label_dist.values():
            p = count / n_total
            if p > 0:
                entropy -= p * math.log2(p)
        shannon_entropy = round(entropy, 4)

        # 4. Confidence Variance
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        confidence_variance = round(variance, 6)

        # 5. Pairwise Agreement Matrix and Score
        agreements = 0
        pairs = 0
        matrix: Dict[str, Dict[str, float]] = {}

        for i, v1 in enumerate(verifications):
            p1_name = v1.provider_name
            if p1_name not in matrix:
                matrix[p1_name] = {}
            for j, v2 in enumerate(verifications):
                p2_name = v2.provider_name
                agree = 1.0 if v1.label == v2.label else 0.0
                matrix[p1_name][p2_name] = agree
                if i < j:
                    pairs += 1
                    if agree == 1.0:
                        agreements += 1

        pairwise_agreement_score = round(agreements / pairs, 4) if pairs > 0 else 1.0

        # 6. Disagreeing Verifiers
        dissenters: List[DisagreementDetail] = []
        for v in verifications:
            if v.label != majority_label:
                dissenters.append(
                    DisagreementDetail(
                        verifier_name=v.provider_name,
                        assigned_label=v.label,
                        confidence=v.confidence,
                        reasoning=v.reasoning,
                    )
                )

        summary = (
            f"Consensus: {majority_label.value} (Confidence: {consensus_confidence:.2f}, "
            f"Agreement: {pairwise_agreement_score*100:.1f}%, Entropy: {shannon_entropy:.2f}). "
            f"{len(dissenters)} dissenting verifier(s)."
        )

        logger.info(
            "consensus_computed",
            claim_id=claim_id,
            majority_label=majority_label.value,
            confidence=consensus_confidence,
            entropy=shannon_entropy,
        )

        return ConsensusResult(
            claim_id=claim_id,
            majority_label=majority_label,
            weighted_label=weighted_label,
            consensus_confidence=consensus_confidence,
            label_distribution=label_dist,
            label_weights=label_weights,
            pairwise_agreement_score=pairwise_agreement_score,
            shannon_entropy=shannon_entropy,
            confidence_variance=confidence_variance,
            agreement_matrix=matrix,
            disagreeing_verifiers=dissenters,
            verdict_summary=summary,
        )
