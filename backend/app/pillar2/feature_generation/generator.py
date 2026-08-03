"""
HalluciSense Pillar 2 — Evidence Feature Generator
===================================================
Computes 10 quantitative evidence-level features for integration with HalluciSense.
"""

import math
from typing import Dict, List

import structlog
from app.pillar2.claim_extraction.schemas import ExtractedClaim
from app.pillar2.consensus_engine.schemas import ConsensusResult
from app.pillar2.evidence_retrieval.schemas import EvidenceItem
from app.pillar2.feature_generation.schemas import PillarTwoFeatures
from app.pillar2.multi_llm_verifier.schemas import VerificationLabel

logger = structlog.get_logger(__name__)


class EvidenceFeatureGenerator:
    """
    Computes 10 evidence feature signals from claims, evidence, and consensus records.
    """

    def generate_features(
        self,
        claims: List[ExtractedClaim],
        evidence_items: List[EvidenceItem],
        consensus_map: Dict[str, ConsensusResult],
    ) -> PillarTwoFeatures:
        """
        Compute PillarTwoFeatures instance.

        Parameters
        ----------
        claims : List[ExtractedClaim]
        evidence_items : List[EvidenceItem]
        consensus_map : Dict[str, ConsensusResult]

        Returns
        -------
        PillarTwoFeatures
        """
        n_claims = len(claims)
        n_ev = len(evidence_items)

        if n_claims == 0:
            return PillarTwoFeatures(
                support_ratio=0.0,
                contradiction_ratio=0.0,
                authority_score=0.0,
                source_diversity=0.0,
                evidence_coverage=0.0,
                evidence_density=0.0,
                citation_quality=0.0,
                consensus_confidence=0.0,
                recency_score=0.0,
                verification_completeness=0.0,
            )

        # 1. Support & Contradiction Ratio
        n_supported = 0
        n_contradicted = 0
        n_covered = 0
        sum_consensus_conf = 0.0

        for claim in claims:
            cid = claim.claim_id
            consensus = consensus_map.get(cid)
            if consensus:
                if consensus.majority_label == VerificationLabel.SUPPORTED:
                    n_supported += 1
                elif consensus.majority_label == VerificationLabel.CONTRADICTED:
                    n_contradicted += 1
                if consensus.majority_label != VerificationLabel.UNKNOWN:
                    n_covered += 1
                sum_consensus_conf += consensus.consensus_confidence

        support_ratio = round(n_supported / n_claims, 4)
        contradiction_ratio = round(n_contradicted / n_claims, 4)
        evidence_coverage = round(n_covered / n_claims, 4)
        consensus_confidence = round(sum_consensus_conf / n_claims, 4)

        # 2. Authority Score & Citation Quality & Recency
        if n_ev > 0:
            authority_score = round(sum(e.authority_score for e in evidence_items) / n_ev, 4)
            citation_quality = round(
                sum(1 for e in evidence_items if e.citation_metadata.doi or e.citation_metadata.journal) / n_ev,
                4,
            )
            recency_score = round(
                sum(1 for e in evidence_items if e.publication_date and any(yr in e.publication_date for yr in ["2021", "2022", "2023", "2024", "2025", "2026"])) / n_ev,
                4,
            )
        else:
            authority_score = 0.0
            citation_quality = 0.0
            recency_score = 0.0

        # 3. Source Diversity (Normalized Shannon Entropy of sources)
        source_counts: Dict[str, int] = {}
        for e in evidence_items:
            source_counts[e.source] = source_counts.get(e.source, 0) + 1

        if len(source_counts) > 1 and n_ev > 0:
            entropy = sum(-(cnt / n_ev) * math.log2(cnt / n_ev) for cnt in source_counts.values())
            max_entropy = math.log2(len(source_counts))
            source_diversity = round(entropy / max_entropy, 4) if max_entropy > 0 else 1.0
        elif len(source_counts) == 1:
            source_diversity = 0.5
        else:
            source_diversity = 0.0

        # 4. Evidence Density & Verification Completeness
        evidence_density = round(n_ev / n_claims, 4)
        completeness = round(min(1.0, (n_covered / n_claims) * 0.7 + (min(n_ev, n_claims * 2) / (n_claims * 2)) * 0.3), 4)

        features = PillarTwoFeatures(
            support_ratio=support_ratio,
            contradiction_ratio=contradiction_ratio,
            authority_score=authority_score,
            source_diversity=source_diversity,
            evidence_coverage=evidence_coverage,
            evidence_density=evidence_density,
            citation_quality=citation_quality,
            consensus_confidence=consensus_confidence,
            recency_score=recency_score,
            verification_completeness=completeness,
        )

        logger.info(
            "pillar2_features_generated",
            support_ratio=support_ratio,
            contradiction_ratio=contradiction_ratio,
            authority_score=authority_score,
            consensus_confidence=consensus_confidence,
        )

        return features
