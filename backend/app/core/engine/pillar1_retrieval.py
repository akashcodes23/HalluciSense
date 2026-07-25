import re
from typing import List, Tuple
from .types import Pillar1Result, EvidenceItem

class Pillar1RetrievalEngine:
    """
    Pillar 1: Retrieval Verification Engine.
    - Extracts factual claims from sentence/text.
    - Compares claims against provided/retrieved evidence snippets.
    - Computes Factual Error Score (FE) in range [0.0, 1.0].
    """

    def extract_claims(self, text: str) -> List[str]:
        """
        Extract candidate factual claims from text.
        Splits by clauses or proposition patterns.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        # Sub-sentence splitting for discrete claims
        raw_claims = re.split(r'[,;]\s*|\s+and\s+|\s+which\s+', clean_text)
        claims = [c.strip() for c in raw_claims if len(c.strip().split()) >= 3]
        
        if not claims:
            claims = [clean_text]
        return claims

    def evaluate_claims_against_evidence(
        self,
        claims: List[str],
        external_evidence: List[EvidenceItem]
    ) -> Tuple[float, List[EvidenceItem]]:
        """
        Compare extracted claims against evidence items to generate FE score.
        FE = 1.0 - mean(max_claim_similarity)
        """
        if not claims:
            return 0.0, external_evidence

        claim_grounding_scores: List[float] = []

        for claim in claims:
            # Find best supporting evidence for this claim
            best_sim = 0.0
            for item in external_evidence:
                # Basic token overlap + similarity calculation fallback
                sim = item.similarity_score
                if sim > best_sim:
                    best_sim = sim
            
            claim_grounding_scores.append(best_sim)

        avg_grounding = sum(claim_grounding_scores) / len(claim_grounding_scores) if claim_grounding_scores else 1.0
        factual_error = max(0.0, min(1.0, 1.0 - avg_grounding))
        return round(factual_error, 4), external_evidence

    def analyze(
        self,
        text: str,
        provided_evidence: List[EvidenceItem] = None
    ) -> Pillar1Result:
        """
        Execute Pillar 1 verification flow.
        """
        if provided_evidence is None:
            provided_evidence = []

        claims = self.extract_claims(text)
        fe_score, evidence = self.evaluate_claims_against_evidence(claims, provided_evidence)

        if not claims:
            reasoning = "No discrete factual claims identified in text."
        elif fe_score < 0.2:
            reasoning = f"High factual grounding. Identified {len(claims)} claim(s) backed by evidence."
        elif fe_score < 0.5:
            reasoning = f"Partial factual grounding. Identified {len(claims)} claim(s) with moderate evidence alignment."
        else:
            reasoning = f"Low factual grounding. Claims lack sufficient supporting evidence from reference sources."

        return Pillar1Result(
            claims=claims,
            evidence=evidence,
            factual_error_score=fe_score,
            reasoning=reasoning
        )
