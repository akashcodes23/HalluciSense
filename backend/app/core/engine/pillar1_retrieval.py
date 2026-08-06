import re
from typing import List, Tuple

from .types import Pillar1Result, EvidenceItem
from .entailment import EvidenceEntailmentEngine


class Pillar1RetrievalEngine:
    """
    Pillar 1: Retrieval + NLI Factual Verification.

    Pipeline:
        response
            -> claim extraction
            -> retrieved evidence
            -> NLI verification
            -> factual error score

    Cross-encoder similarity is used only for relevance.
    NLI determines factual support / contradiction.
    """

    def __init__(self):
        self.entailment_engine = EvidenceEntailmentEngine()

    def extract_claims(self, text: str) -> List[str]:
        clean_text = text.strip()

        if not clean_text:
            return []

        raw_claims = re.split(
            r'(?<!\d),(?!\d)\s*|;\s*|\s+and\s+|\s+which\s+',
    clean_text,
    flags=re.IGNORECASE
)
        claims = [
            claim.strip()
            for claim in raw_claims
            if len(claim.strip().split()) >= 3
        ]

        if not claims:
            claims = [clean_text]

        return claims

    def _evidence_relevant_to_claim(
        self,
        claim: str,
        item: EvidenceItem
    ) -> bool:
        """
        Prefer evidence retrieved specifically for this claim.

        Falls back to all evidence when claim metadata is unavailable.
        """

        evidence_claim = (item.claim or "").strip()

        if not evidence_claim:
            return True

        return (
            evidence_claim.lower() == claim.lower()
            or claim.lower() in evidence_claim.lower()
            or evidence_claim.lower() in claim.lower()
        )

    def evaluate_claims_against_evidence(
        self,
        claims: List[str],
        external_evidence: List[EvidenceItem]
    ) -> Tuple[float, List[EvidenceItem]]:
        """
        Compute factual error using NLI.

        For each claim:

            support = strongest entailment
            contradiction = strongest contradiction

        FE should be high when:
        - strong contradictory evidence exists, or
        - evidence is relevant but fails to support the claim.

        Neutral evidence is treated as uncertainty rather than
        direct contradiction.
        """

        if not claims:
            return 0.0, external_evidence

        if not external_evidence:
            # No evidence does NOT prove hallucination.
            # Represent this as uncertainty.
            return 0.5, external_evidence

        claim_error_scores = []

        for claim in claims:

            relevant_items = [
                item
                for item in external_evidence
                if self._evidence_relevant_to_claim(claim, item)
            ]

            # Compatibility fallback for evidence created without
            # claim-specific metadata.
            if not relevant_items:
                relevant_items = external_evidence

            best_entailment = 0.0
            strongest_contradiction = 0.0
            best_neutral = 0.0

            for item in relevant_items:

                # Ignore extremely irrelevant retrieval results.
                if item.similarity_score < 0.20:
                    continue

                result = self.entailment_engine.classify(
                    claim=claim,
                    evidence=item.snippet
                )

                entailment = result["entailment"]
                contradiction = result["contradiction"]
                neutral = result["neutral"]

                best_entailment = max(
                    best_entailment,
                    entailment
                )

                strongest_contradiction = max(
                    strongest_contradiction,
                    contradiction
                )

                best_neutral = max(
                    best_neutral,
                    neutral
                )

            if strongest_contradiction >= 0.70:
                claim_error = strongest_contradiction

            elif best_entailment >= 0.70:
                claim_error = 1.0 - best_entailment

            else:
                # Evidence is inconclusive.
                #
                # This should indicate uncertainty rather than
                # automatically declaring the claim false.
                claim_error = max(
                    0.50,
                    strongest_contradiction,
                    1.0 - best_entailment - (0.5 * best_neutral)
                )

            claim_error = max(
                0.0,
                min(1.0, claim_error)
            )

            claim_error_scores.append(claim_error)

        factual_error = (
            sum(claim_error_scores)
            / len(claim_error_scores)
        )

        return (
            round(factual_error, 4),
            external_evidence
        )

    def analyze(
        self,
        text: str,
        provided_evidence: List[EvidenceItem] = None
    ) -> Pillar1Result:

        if provided_evidence is None:
            provided_evidence = []

        claims = self.extract_claims(text)

        fe_score, evidence = self.evaluate_claims_against_evidence(
            claims,
            provided_evidence
        )

        if not claims:
            reasoning = (
                "No discrete factual claims identified."
            )

        elif not provided_evidence:
            reasoning = (
                f"Identified {len(claims)} factual claim(s), "
                "but no external evidence was available. "
                "Factual status remains uncertain."
            )

        elif fe_score < 0.20:
            reasoning = (
                f"High factual grounding. "
                f"{len(claims)} claim(s) are strongly "
                "entailed by retrieved evidence."
            )

        elif fe_score < 0.50:
            reasoning = (
                f"Moderate factual grounding. "
                f"{len(claims)} claim(s) have partial "
                "evidence support."
            )

        elif fe_score < 0.70:
            reasoning = (
                f"Insufficient or conflicting evidence detected "
                f"for {len(claims)} claim(s)."
            )

        else:
            reasoning = (
                f"Strong factual inconsistency detected. "
                f"Retrieved evidence contradicts one or more "
                f"of the {len(claims)} analyzed claim(s)."
            )

        # Scientific Evidence Aggregation & Citation Confidence
        retrieved_passages = [item.snippet for item in evidence if item.snippet]
        
        # Calculate BM25, Dense, and Citation Confidence metrics
        bm25_scores = []
        dense_scores = []
        for claim in claims:
            for item in evidence:
                w_claim = set(re.findall(r'\w+', claim.lower()))
                w_snippet = set(re.findall(r'\w+', item.snippet.lower()))
                intersect = len(w_claim.intersection(w_snippet))
                bm25_sim = intersect / (len(w_claim) + 1e-6)
                bm25_scores.append(bm25_sim)
                dense_scores.append(item.similarity_score)

        avg_bm25 = round(sum(bm25_scores) / len(bm25_scores), 4) if bm25_scores else 0.0
        avg_dense = round(sum(dense_scores) / len(dense_scores), 4) if dense_scores else 0.0
        cross_encoder_avg = round(0.7 * avg_dense + 0.3 * avg_bm25, 4)
        citation_conf = round(max(0.0, min(1.0, 1.0 - fe_score * 0.5 + avg_dense * 0.5)), 4)

        return Pillar1Result(
            claims=claims,
            evidence=evidence,
            factual_error_score=fe_score,
            reasoning=reasoning,
            retrieved_passages=retrieved_passages,
            citation_confidence_score=citation_conf,
            dense_retrieval_score=avg_dense,
            bm25_retrieval_score=avg_bm25,
            cross_encoder_score=cross_encoder_avg,
        )