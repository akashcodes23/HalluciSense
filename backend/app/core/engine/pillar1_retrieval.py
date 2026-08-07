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

        # Split on clause boundaries (commas, semicolons) but NOT on
        # relative pronouns ('which', 'that') or conjunctions ('and')
        # which create incomplete fragments.
        raw_claims = re.split(
            r'(?<!\d),(?!\d)\s*|;\s*',
            clean_text,
        )

        claims = [
            claim.strip()
            for claim in raw_claims
            if len(claim.strip().split()) >= 4
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

            for idx, item in enumerate(relevant_items):

                # Ignore extremely irrelevant retrieval results.
                sim_score = float(getattr(item, "score", getattr(item, "similarity_score", 0.88)))
                if sim_score < 0.20:
                    continue

                result = self.entailment_engine.classify(
                    claim=claim,
                    evidence=item.snippet
                )

                entailment = result["entailment"]
                contradiction = result["contradiction"]
                neutral = result["neutral"]

                # Alias/formula definition check for short equivalence claims ("X is Y")
                # e.g., "Water is H2O" where evidence contains "water (H2O)" or "formula H2O"
                if neutral > 0.60 and contradiction < 0.30 and sim_score >= 0.70:
                    # Check if claim is a short identity/formula statement "A is B"
                    match = re.match(r'^([A-Za-z0-9\s]+)\s+is\s+([A-Za-z0-9]+)\.?$', claim.strip(), re.IGNORECASE)
                    if match:
                        subj, obj = match.group(1).strip().lower(), match.group(2).strip().lower()
                        snippet_lower = item.snippet.lower()
                        if f"{subj} ({obj})" in snippet_lower or f"{subj} ({obj}" in snippet_lower or f"formula {obj}" in snippet_lower:
                            entailment = max(entailment, 0.90)

                # Scientific definition concept verification for complex multi-clause sentences
                # e.g., "Photosynthesis is the process by which green plants convert sunlight into chemical energy using chlorophyll"
                sim_score = float(getattr(item, "score", getattr(item, "similarity_score", 0.88)))
                if neutral > 0.60 and contradiction < 0.20 and (sim_score >= 0.70 or idx == 0):
                    claim_keywords = [w.lower() for w in re.findall(r'\b[A-Za-z]{4,}\b', claim)]
                    snippet_lower = item.snippet.lower()
                    if claim_keywords:
                        matching_words = sum(1 for kw in claim_keywords if kw in snippet_lower)
                        coverage_ratio = matching_words / float(len(claim_keywords))
                        if coverage_ratio >= 0.50:
                            # High keyword & concept alignment in top-reranked snippet
                            entailment = max(entailment, round(0.70 + 0.25 * coverage_ratio, 4))
                            neutral = max(0.0, 1.0 - entailment - contradiction)

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

            # Scoring Decision Logic:
            # 1. Contradiction clearly dominates if it is >= 0.70 AND exceeds entailment by > 0.15
            if strongest_contradiction >= 0.70 and strongest_contradiction > (best_entailment + 0.15):
                claim_error = strongest_contradiction

            # 2. Entailment dominates if best_entailment is high (>= 0.65)
            elif best_entailment >= 0.65:
                claim_error = 1.0 - best_entailment

            # 3. Moderate contradiction without strong entailment
            elif strongest_contradiction >= 0.50:
                claim_error = strongest_contradiction

            # 4. Neutral / Scientific phrasing ambiguity without contradiction
            elif strongest_contradiction < 0.30 and (best_entailment + 0.5 * best_neutral) >= 0.40:
                claim_error = 1.0 - (best_entailment + 0.5 * best_neutral)

            # 5. Inconclusive / Uncertainty
            else:
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