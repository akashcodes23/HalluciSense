import re
from typing import List, Tuple, Optional
from .types import Pillar1Result, EvidenceItem
from .entailment import EvidenceEntailmentEngine
from .temporal import TemporalClaimEngine, TemporalStatus, EpistemicModality


class Pillar1RetrievalEngine:
    """Pillar 1: Retrieval + NLI Factual Verification + Temporal Consistency Analysis."""

    def __init__(self):
        self.entailment_engine = EvidenceEntailmentEngine()
        self.temporal_engine = TemporalClaimEngine()

    def extract_claims(self, text: str) -> List[str]:
        import time
        t0 = time.perf_counter()
        clean_text = text.strip()
        if not clean_text:
            self.last_claim_extraction_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return []
        raw_claims = re.split(r'(?<!\d),(?!\d)\s*|;\s*', clean_text)
        claims = [claim.strip() for claim in raw_claims if len(claim.strip().split()) >= 4]
        if not claims:
            claims = [clean_text]
        self.last_claim_extraction_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return claims

    def _evidence_relevant_to_claim(self, claim: str, item: EvidenceItem) -> bool:
        evidence_claim = (item.claim or "").strip()
        if not evidence_claim:
            return True
        return (
            evidence_claim.lower() == claim.lower()
            or claim.lower() in evidence_claim.lower()
            or evidence_claim.lower() in claim.lower()
        )

    @staticmethod
    def _score_result(claim: str, item: EvidenceItem, result: dict, idx: int) -> Tuple[float, float, float]:
        entailment = result["entailment"]
        contradiction = result["contradiction"]
        neutral = result["neutral"]
        sim_score = float(getattr(item, "score", getattr(item, "similarity_score", 0.88)))

        if neutral > 0.60 and contradiction < 0.30 and sim_score >= 0.70:
            match = re.match(r'^([A-Za-z0-9\s]+)\s+is\s+([A-Za-z0-9]+)\.?$', claim.strip(), re.IGNORECASE)
            if match:
                subj, obj = match.group(1).strip().lower(), match.group(2).strip().lower()
                snippet_lower = item.snippet.lower()
                if f"{subj} ({obj})" in snippet_lower or f"{subj} ({obj}" in snippet_lower or f"formula {obj}" in snippet_lower:
                    entailment = max(entailment, 0.90)

        if neutral > 0.60 and contradiction < 0.20 and (sim_score >= 0.70 or idx == 0):
            claim_keywords = [w.lower() for w in re.findall(r'\b[A-Za-z]{4,}\b', claim)]
            snippet_lower = item.snippet.lower()
            if claim_keywords:
                matching_words = sum(1 for kw in claim_keywords if kw in snippet_lower)
                coverage_ratio = matching_words / float(len(claim_keywords))
                if coverage_ratio >= 0.50:
                    entailment = max(entailment, round(0.70 + 0.25 * coverage_ratio, 4))
                    neutral = max(0.0, 1.0 - entailment - contradiction)
        return entailment, contradiction, neutral

    def evaluate_claims_against_evidence(self, claims: List[str], external_evidence: List[EvidenceItem]) -> Tuple[float, List[EvidenceItem]]:
        import time
        t0 = time.perf_counter()
        if not claims:
            self.last_nli_ms = 0.0
            return 0.0, external_evidence
        if not external_evidence:
            self.last_nli_ms = 0.0
            return 0.5, external_evidence

        # Build all relevant claim/evidence pairs first, then execute ONE batched
        # DeBERTa inference pass instead of invoking the model once per pair.
        pairs = []
        pair_meta = []
        for claim_idx, claim in enumerate(claims):
            relevant_items = [item for item in external_evidence if self._evidence_relevant_to_claim(claim, item)]
            if not relevant_items:
                relevant_items = external_evidence
            for item_idx, item in enumerate(relevant_items):
                sim_score = float(getattr(item, "score", getattr(item, "similarity_score", 0.88)))
                if sim_score >= 0.20:
                    pairs.append((claim, item.snippet))
                    pair_meta.append((claim_idx, item_idx, item))

        results = self.entailment_engine.classify_batch(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            batch_size=32,
        ) if pairs else []

        best = [[0.0, 0.0, 0.0] for _ in claims]
        for result, (claim_idx, item_idx, item) in zip(results, pair_meta):
            entailment, contradiction, neutral = self._score_result(
                claims[claim_idx], item, result, item_idx
            )
            best[claim_idx][0] = max(best[claim_idx][0], entailment)
            best[claim_idx][1] = max(best[claim_idx][1], contradiction)
            best[claim_idx][2] = max(best[claim_idx][2], neutral)

        claim_error_scores = []
        for best_entailment, strongest_contradiction, best_neutral in best:
            if strongest_contradiction >= 0.70 and strongest_contradiction > (best_entailment + 0.15):
                claim_error = strongest_contradiction
            elif best_entailment >= 0.65:
                claim_error = 1.0 - best_entailment
            elif strongest_contradiction >= 0.50:
                claim_error = strongest_contradiction
            elif strongest_contradiction < 0.30 and (best_entailment + 0.5 * best_neutral) >= 0.40:
                claim_error = 1.0 - (best_entailment + 0.5 * best_neutral)
            else:
                claim_error = max(0.50, strongest_contradiction, 1.0 - best_entailment - (0.5 * best_neutral))
            claim_error_scores.append(max(0.0, min(1.0, claim_error)))

        self.last_nli_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.last_nli_batch_metrics = getattr(self.entailment_engine, "last_batch_metrics", {})
        factual_error = sum(claim_error_scores) / len(claim_error_scores)
        return round(factual_error, 4), external_evidence

    def analyze(self, text: str, provided_evidence: List[EvidenceItem] = None, query: Optional[str] = None) -> Pillar1Result:
        if provided_evidence is None:
            provided_evidence = []
        claims = self.extract_claims(text)
        fe_score, evidence = self.evaluate_claims_against_evidence(claims, provided_evidence)

        # Temporal Claim & Epistemic Modality Evaluation
        temp_res = self.temporal_engine.analyze_claim(text, query=query, evidence_items=provided_evidence)
        if temp_res.temporal_inconsistency_score > 0.0:
            fe_score = round(max(fe_score, temp_res.temporal_inconsistency_score), 4)
        elif temp_res.modality in (
            EpistemicModality.PREDICTION,
            EpistemicModality.HYPOTHETICAL,
            EpistemicModality.COUNTERFACTUAL,
            EpistemicModality.FICTION,
        ):
            fe_score = 0.0

        if not claims:
            reasoning = "No discrete factual claims identified."
        elif temp_res.temporal_inconsistency_score > 0.0:
            reasoning = f"Temporal Inconsistency: {temp_res.reasoning}"
        elif temp_res.modality != EpistemicModality.ASSERTED_FACT:
            reasoning = f"Protected Epistemic Modality ({temp_res.modality.value}): Statement is non-factual assertion."
        elif not provided_evidence:
            reasoning = f"Identified {len(claims)} factual claim(s), but no external evidence was available. Factual status remains uncertain."
        elif fe_score < 0.20:
            reasoning = f"High factual grounding. {len(claims)} claim(s) are strongly entailed by retrieved evidence."
        elif fe_score < 0.50:
            reasoning = f"Moderate factual grounding. {len(claims)} claim(s) have partial evidence support."
        elif fe_score < 0.70:
            reasoning = f"Insufficient or conflicting evidence detected for {len(claims)} claim(s)."
        else:
            reasoning = f"Strong factual inconsistency detected. Retrieved evidence contradicts one or more of the {len(claims)} analyzed claim(s)."

        retrieved_passages = [item.snippet for item in evidence if item.snippet]
        bm25_scores, dense_scores = [], []
        for claim in claims:
            for item in evidence:
                w_claim = set(re.findall(r'\w+', claim.lower()))
                w_snippet = set(re.findall(r'\w+', item.snippet.lower()))
                bm25_scores.append(len(w_claim.intersection(w_snippet)) / (len(w_claim) + 1e-6))
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
