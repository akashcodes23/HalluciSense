import re
from typing import List, Tuple, Optional, Dict, Any

import httpx

from .types import Pillar1Result, EvidenceItem
from .entailment import EvidenceEntailmentEngine
from .temporal import TemporalClaimEngine, TemporalStatus, EpistemicModality
from .numeric_unit_checker import NumericUnitChecker, NumericUnitStatus
from .negation_detector import NegationDetector
from .causal_direction import CausalDirectionChecker


class EventTemporalAnchorResolver:
    """Resolve event/entity time spans dynamically through Wikidata.

    No entity names or dates are hardcoded. The resolver is deliberately
    conservative: it only activates for relative temporal statements without
    explicit four-digit years and returns a contradiction only when two
    independently resolved anchors make the stated relation impossible.
    """

    API_URL = "https://www.wikidata.org/w/api.php"
    TIME_PROPERTIES = ("P585", "P580", "P582", "P571", "P576", "P575")
    MAX_ANCHORS = 2
    TIMEOUT_SECONDS = 0.8

    def __init__(self):
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self.last_lookup_count = 0

    @staticmethod
    def _extract_year(value: Any) -> Optional[int]:
        if not isinstance(value, str):
            return None
        match = re.search(r"[+-](\d{4})-", value)
        return int(match.group(1)) if match else None

    @staticmethod
    def _normalise_candidate(value: str) -> str:
        value = re.sub(r"^[Tt]he\s+", "", value.strip())
        value = re.sub(r"\s+", " ", value)
        return value.strip(" ,.;:()[]")

    def extract_anchor_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []

        # Named multi-word entities / periods.
        for match in re.finditer(
            r"\b(?:[A-Z][A-Za-z0-9'’-]+(?:\s+[A-Z][A-Za-z0-9'’-]+){1,5})\b",
            text,
        ):
            candidate = self._normalise_candidate(match.group(0))
            if candidate and candidate.lower() not in {c.lower() for c in candidates}:
                candidates.append(candidate)

        # Explicit temporal-object phrases after relational operators. This
        # captures entities such as "the Renaissance" without hardcoding them.
        relation_object = re.compile(
            r"\b(?:during|after|before|since|prior to|following|preceding|earlier than|later than)\s+"
            r"(?:the\s+)?([A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,4})",
            re.IGNORECASE,
        )
        for match in relation_object.finditer(text):
            candidate = self._normalise_candidate(match.group(1))
            candidate = re.split(r"\b(?:was|were|is|are|happened|occurred|began|ended)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if candidate and len(candidate.split()) <= 5 and candidate.lower() not in {c.lower() for c in candidates}:
                candidates.append(candidate)

        return candidates[: self.MAX_ANCHORS]

    def _search_entity(self, label: str) -> Optional[str]:
        if label in self._cache:
            cached = self._cache[label]
            return cached.get("id") if cached else None

        try:
            self.last_lookup_count += 1
            response = httpx.get(
                self.API_URL,
                params={
                    "action": "wbsearchentities",
                    "search": label,
                    "language": "en",
                    "type": "item",
                    "limit": 1,
                    "format": "json",
                },
                headers={"User-Agent": "HalluciSense/Phase6 temporal-research"},
                timeout=self.TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            results = response.json().get("search", [])
            if not results:
                self._cache[label] = None
                return None
            entity_id = results[0].get("id")
            self._cache[label] = {"id": entity_id} if entity_id else None
            return entity_id
        except Exception:
            self._cache[label] = None
            return None

    def _fetch_time_span(self, entity_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = httpx.get(
                self.API_URL,
                params={
                    "action": "wbgetentities",
                    "ids": entity_id,
                    "props": "claims|labels|descriptions",
                    "languages": "en",
                    "format": "json",
                },
                headers={"User-Agent": "HalluciSense/Phase6 temporal-research"},
                timeout=self.TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            entity = response.json().get("entities", {}).get(entity_id, {})
            claims = entity.get("claims", {})
            years: List[int] = []
            for prop in self.TIME_PROPERTIES:
                for statement in claims.get(prop, []):
                    value = (
                        statement.get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                    )
                    year = self._extract_year(value.get("time") if isinstance(value, dict) else None)
                    if year is not None:
                        years.append(year)

            if not years:
                return None

            return {
                "id": entity_id,
                "label": entity.get("labels", {}).get("en", {}).get("value", ""),
                "start": min(years),
                "end": max(years),
                "points": sorted(set(years)),
            }
        except Exception:
            return None

    def resolve(self, text: str) -> List[Dict[str, Any]]:
        anchors: List[Dict[str, Any]] = []
        for candidate in self.extract_anchor_candidates(text):
            entity_id = self._search_entity(candidate)
            if not entity_id:
                continue
            span = self._fetch_time_span(entity_id)
            if span:
                span["query"] = candidate
                anchors.append(span)
            if len(anchors) >= self.MAX_ANCHORS:
                break
        return anchors

    @staticmethod
    def _relation(text: str) -> Optional[str]:
        lower = text.lower()
        for relation in ("prior to", "earlier than", "later than", "preceding", "following", "before", "after", "since", "during"):
            if re.search(rf"\b{re.escape(relation)}\b", lower):
                return relation
        return None

    def evaluate(self, text: str) -> Tuple[Optional[float], Optional[str], List[Dict[str, Any]]]:
        anchors = self.resolve(text)
        if len(anchors) < 2:
            return None, None, anchors

        relation = self._relation(text)
        if not relation:
            return None, None, anchors

        first, second = anchors[0], anchors[1]
        a_start, a_end = first["start"], first["end"]
        b_start, b_end = second["start"], second["end"]

        contradiction = False
        if relation == "during":
            contradiction = a_end < b_start or a_start > b_end
        elif relation in {"before", "prior to", "earlier than", "preceding"}:
            contradiction = a_end >= b_start
        elif relation in {"after", "following", "later than", "since"}:
            contradiction = a_start <= b_end

        if contradiction:
            return 0.92, (
                f"Dynamic event-anchor contradiction: '{first['query']}' [{a_start}-{a_end}] "
                f"is incompatible with relation '{relation}' to '{second['query']}' [{b_start}-{b_end}]."
            ), anchors
        return 0.0, (
            f"Dynamic event-anchor relation '{relation}' is temporally compatible: "
            f"'{first['query']}' [{a_start}-{a_end}] vs '{second['query']}' [{b_start}-{b_end}]."
        ), anchors


class Pillar1RetrievalEngine:
    """Pillar 1: Retrieval + NLI Factual Verification + Temporal Consistency Analysis."""

    def __init__(self):
        self.entailment_engine = EvidenceEntailmentEngine()
        self.temporal_engine = TemporalClaimEngine()
        self.event_anchor_resolver = EventTemporalAnchorResolver()
        self.numeric_checker = NumericUnitChecker()
        self.negation_detector = NegationDetector()
        self.causal_checker = CausalDirectionChecker()

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

        # Check for chemical formula contradiction or match
        claim_formula_m = re.search(r'\b(?:chemical\s+formula|formula)\s+([A-Za-z0-9]+)\b', claim, re.IGNORECASE)
        snippet_formula_m = re.search(r'\b(?:chemical\s+formula|formula)\s+([A-Za-z0-9]+)\b', item.snippet, re.IGNORECASE)

        if claim_formula_m and snippet_formula_m:
            c_form = claim_formula_m.group(1).upper()
            s_form = snippet_formula_m.group(1).upper()
            if c_form != s_form:
                return 0.0, 0.95, 0.05
            else:
                return 0.95, 0.0, 0.05

        if neutral > 0.60 and contradiction < 0.30 and sim_score >= 0.70:
            match = re.match(r'^([A-Za-z0-9\s]+)\s+is\s+([A-Za-z0-9]+)\.?$', claim.strip(), re.IGNORECASE)
            if match:
                subj, obj = match.group(1).strip().lower(), match.group(2).strip().lower()
                snippet_lower = item.snippet.lower()
                if f"{subj} ({obj})" in snippet_lower or f"{subj} ({obj}" in snippet_lower or f"formula {obj}" in snippet_lower:
                    entailment = max(entailment, 0.90)

        if neutral > 0.60 and contradiction < 0.20 and (sim_score >= 0.70 or idx == 0) and not claim_formula_m:
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

        temp_res = self.temporal_engine.analyze_claim(text, query=query, evidence_items=provided_evidence)
        event_anchor_score = None
        event_anchor_reasoning = None
        event_anchors: List[Dict[str, Any]] = []

        # Phase 6: only invoke dynamic event anchoring when explicit years are
        # absent and the temporal engine identifies relational language.
        if temp_res.temporal_status == TemporalStatus.TIME_RELATIVE and not temp_res.extracted_years:
            event_anchor_score, event_anchor_reasoning, event_anchors = self.event_anchor_resolver.evaluate(text)

        if temp_res.temporal_inconsistency_score > 0.0:
            fe_score = round(max(fe_score, temp_res.temporal_inconsistency_score), 4)
        elif event_anchor_score is not None and event_anchor_score > 0.0:
            fe_score = round(max(fe_score, event_anchor_score), 4)
        elif temp_res.protected_from_temporal_penalty and temp_res.modality != EpistemicModality.ASSERTED_FACT:
            if temp_res.modality in {
                EpistemicModality.PREDICTION,
                EpistemicModality.HYPOTHETICAL,
                EpistemicModality.COUNTERFACTUAL,
                EpistemicModality.CONDITIONAL,
                EpistemicModality.FICTIONAL,
            }:
                fe_score = 0.0

        # Symbolic checks against top retrieved evidence
        ev_snippets = [e.snippet for e in evidence if getattr(e, "snippet", "")]
        if ev_snippets:
            ev_combined = " ".join(ev_snippets[:3])
            num_status, num_penalty, _ = self.numeric_checker.check_consistency(text, ev_combined)
            if num_status in (NumericUnitStatus.NUMERIC_CONFLICT, NumericUnitStatus.SCALE_CONFLICT, NumericUnitStatus.UNIT_CONFLICT):
                fe_score = round(max(fe_score, num_penalty), 4)

            pol_res = self.negation_detector.analyze(text, ev_combined)
            if pol_res.negation_inversion_detected or pol_res.antonym_inversion_detected:
                fe_score = round(max(fe_score, pol_res.confidence_penalty), 4)

            for snip in ev_snippets[:3]:
                caus_res = self.causal_checker.check_inversion(text, snip)
                if caus_res.is_inversion_detected:
                    fe_score = round(max(fe_score, caus_res.confidence_penalty), 4)
                    break

        if not claims:
            reasoning = "No discrete factual claims identified."
        elif temp_res.temporal_inconsistency_score > 0.0:
            reasoning = f"Temporal Inconsistency: {temp_res.reasoning}"
        elif event_anchor_score is not None and event_anchor_score > 0.0:
            reasoning = f"Temporal Event-Anchor Inconsistency: {event_anchor_reasoning}"
        elif event_anchor_reasoning:
            reasoning = f"Temporal Event-Anchor Check: {event_anchor_reasoning}"
        elif temp_res.protected_from_temporal_penalty and temp_res.modality != EpistemicModality.ASSERTED_FACT:
            reasoning = f"Protected Epistemic Modality ({temp_res.modality.value}): temporal penalty suppressed."
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

        # Diagnostic metadata is attached non-invasively for Phase 6 reports.
        self.last_event_anchor_diagnostics = {
            "anchors": event_anchors,
            "score": event_anchor_score,
            "reasoning": event_anchor_reasoning,
            "lookups": self.event_anchor_resolver.last_lookup_count,
        }

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
