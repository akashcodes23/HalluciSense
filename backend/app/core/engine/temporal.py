"""Temporal Claim Analysis Engine for HalluciSense (Phase 6 architectural remediation).

Phase 6 changes:
- query and response modality are resolved independently;
- evidence-date mismatch is evaluated globally across relevant evidence;
- relational temporal operators and multi-word modality markers are supported;
- metalinguistic/quoted claims are distinguished from ordinary negated facts;
- no-year temporal contradictions can be resolved from dynamically retrieved evidence
  anchors without hard-coded entity dates.

Production fusion weights and risk thresholds are intentionally outside this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TemporalStatus(str, Enum):
    PAST_FACT = "PAST_FACT"
    PRESENT_STATE = "PRESENT_STATE"
    FUTURE_IMPOSSIBLE_FACT = "FUTURE_IMPOSSIBLE_FACT"
    FUTURE_PREDICTION = "FUTURE_PREDICTION"
    HYPOTHETICAL = "HYPOTHETICAL"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    CONDITIONAL = "CONDITIONAL"
    NEGATED_FACT = "NEGATED_FACT"
    FICTIONAL = "FICTIONAL"
    DATE_MISMATCH = "DATE_MISMATCH"
    DATE_RANGE = "DATE_RANGE"
    TIME_RELATIVE = "TIME_RELATIVE"
    UNKNOWN = "UNKNOWN"


class EpistemicModality(str, Enum):
    ASSERTED_FACT = "ASSERTED_FACT"
    FUTURE_FACT_ASSERTION = "FUTURE_FACT_ASSERTION"
    PREDICTION = "PREDICTION"
    HYPOTHETICAL = "HYPOTHETICAL"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    CONDITIONAL = "CONDITIONAL"
    NEGATED_FACT = "NEGATED_FACT"
    FICTIONAL = "FICTIONAL"
    QUOTED_CLAIM = "QUOTED_CLAIM"
    UNKNOWN = "UNKNOWN"


@dataclass
class TemporalAnalysisResult:
    has_temporal_expression: bool
    extracted_years: List[int]
    modality: EpistemicModality
    temporal_status: TemporalStatus
    temporal_inconsistency_score: float
    protected_from_temporal_penalty: bool
    reasoning: str
    query_modality: EpistemicModality = EpistemicModality.UNKNOWN
    response_modality: EpistemicModality = EpistemicModality.UNKNOWN


class TemporalClaimEngine:
    """Generalizable, deterministic context-aware temporal claim analysis."""

    CURRENT_YEAR: int = 2026
    YEAR_PATTERN = re.compile(r"\b(1\d{3}|20\d{2}|2100)\b")

    PREDICTION_PATTERNS = [
        re.compile(r"\b(?:will|shall|is|are|was|were)?\s*(?:expected|predicted|projected|forecast|estimated|scheduled)\b(?:\s+\w+){0,8}\s+\bto\b"),
        re.compile(r"\b(?:plans?|aims?|anticipates?|expects?)\s+to\b"),
        re.compile(r"\b(?:according to|under)\b[^.]{0,80}\b(?:projection|forecast)\b"),
    ]
    HYPOTHETICAL_PATTERNS = [
        re.compile(r"\b(?:suppose|supposing|assuming|hypothetically|imagine|what if)\b"),
        re.compile(r"\bin a scenario where\b"),
        re.compile(r"\bin the event that\b"),
    ]
    COUNTERFACTUAL_PATTERNS = [
        re.compile(r"\bif\b[^.]{0,100}\b(?:had|hadn't|were|wasn't)\b"),
        re.compile(r"\b(?:would have|could have|might have)\b"),
        re.compile(r"\bhad (?:been|won|released|occurred|happened)\b"),
    ]
    CONDITIONAL_PATTERN = re.compile(r"\bif\b[^.]{0,120}\b(?:would|could|might|then)\b")

    FICTION_PATTERNS = [
        re.compile(r"\b(?:in the|within the)\s+(?:fictional\s+)?(?:[\w-]+\s+){0,5}(?:universe|novel|movie|film|story|game)\b"),
        re.compile(r"\b(?:fictional|sci-fi|science fiction|fantasy|mythological|comic book|video game|anime)\b"),
        re.compile(r"\bin fiction\b"),
    ]
    QUOTED_PATTERNS = [
        re.compile(r"\b(?:claim|claims|claimed|claiming|report|reported|reports|rumou?r|rumou?red|post|posts|article|articles|press release|viral post)\b[^.]{0,100}\b(?:that|to)\b"),
        re.compile(r"\b(?:falsely|erroneously|incorrectly|wrongly)\s+(?:reported|claimed|stated|asserted)\b"),
        re.compile(r"\b(?:debunked|discredited|refuted)\s+(?:claim|claims|assertion|assertions|rumou?r)\b"),
        re.compile(r"\bthe claim that\b"),
    ]
    NEGATION_PATTERNS = [
        re.compile(r"\b(?:did|does|do|has|have|had|is|are|was|were)\s+not\b"),
        re.compile(r"\b(?:never|no evidence|no human|nobody|nothing)\b"),
        re.compile(r"\b(?:false|untrue)\s+that\b"),
    ]

    PAST_ACTION_VERBS = [
        "won", "released", "discovered", "defeated", "built", "launched", "created",
        "invented", "died", "signed", "passed", "elected", "hosted", "occurred",
        "took place", "ended", "completed", "published", "landed", "collapsed",
        "patented", "opened", "developed", "declared", "founded", "introduced",
    ]

    RELATION_PATTERNS = [
        ("before", re.compile(r"\b(?:before|prior to|earlier than|preceded|preceding)\b")),
        ("after", re.compile(r"\b(?:after|following|later than|succeeded|subsequent to)\b")),
        ("during", re.compile(r"\b(?:during|within|in the period of|while)\b")),
    ]
    RELATIVE_MARKERS = re.compile(
        r"\b(?:before|after|since|following|prior to|earlier than|later than|preceded|succeeded|during|within)\b"
    )

    _STOPWORDS = {
        "about", "after", "before", "during", "from", "into", "that", "this", "then", "than",
        "there", "their", "they", "were", "with", "what", "when", "where", "which", "would",
        "could", "should", "have", "has", "had", "been", "being", "will", "does", "did", "the",
        "and", "for", "was", "are", "is", "not", "who", "how", "why", "when", "year", "years",
    }

    def _match_any(self, patterns: List[re.Pattern], text: str) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    def _detect_response_modality(self, text: str, claim_text: Optional[str] = None) -> EpistemicModality:
        candidate = (claim_text or text or "").lower().strip()
        if self._match_any(self.FICTION_PATTERNS, candidate):
            return EpistemicModality.FICTIONAL
        if self._match_any(self.QUOTED_PATTERNS, candidate):
            return EpistemicModality.QUOTED_CLAIM
        if self._match_any(self.COUNTERFACTUAL_PATTERNS, candidate):
            return EpistemicModality.COUNTERFACTUAL
        if self._match_any(self.HYPOTHETICAL_PATTERNS, candidate):
            return EpistemicModality.HYPOTHETICAL
        if self.CONDITIONAL_PATTERN.search(candidate) or re.search(r"^\s*if\b", candidate):
            return EpistemicModality.CONDITIONAL
        if self._match_any(self.PREDICTION_PATTERNS, candidate):
            return EpistemicModality.PREDICTION
        if self._match_any(self.NEGATION_PATTERNS, candidate):
            return EpistemicModality.NEGATED_FACT
        return EpistemicModality.ASSERTED_FACT

    def detect_modality(
        self,
        query: str,
        text: str,
        claim_text: Optional[str] = None,
    ) -> EpistemicModality:
        """Backward-compatible API: return the response/claim modality only.

        Phase 6 deliberately does not let query modality leak into the response.
        """
        return self._detect_response_modality(text, claim_text)

    def detect_query_modality(self, query: str) -> EpistemicModality:
        return self._detect_response_modality(query or "")

    def detect_response_modality(self, text: str, claim_text: Optional[str] = None) -> EpistemicModality:
        return self._detect_response_modality(text, claim_text)

    @classmethod
    def _content_tokens(cls, text: str) -> set:
        return {
            token for token in re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", (text or "").lower())
            if token not in cls._STOPWORDS
        }

    @classmethod
    def _evidence_relevance(cls, claim: str, snippet: str, evidence_claim: str = "") -> float:
        claim_tokens = cls._content_tokens(f"{claim} {evidence_claim}")
        snippet_tokens = cls._content_tokens(snippet)
        if not claim_tokens or not snippet_tokens:
            return 0.0
        overlap = len(cls._content_tokens(claim).intersection(snippet_tokens))
        claim_entity_overlap = len(cls._content_tokens(evidence_claim).intersection(cls._content_tokens(claim)))
        return float(max(overlap, claim_entity_overlap))

    @staticmethod
    def _get_evidence_fields(item: Any) -> Tuple[str, str, float, bool]:
        if isinstance(item, dict):
            snippet = item.get("snippet", "") or item.get("retrieved_passage", "") or ""
            claim = item.get("claim", "") or ""
            score = float(item.get("similarity_score", item.get("score", 0.0)) or 0.0)
            supporting = bool(item.get("is_supporting", True))
            return snippet, claim, score, supporting
        snippet = getattr(item, "snippet", None) or getattr(item, "retrieved_passage", None) or ""
        claim = getattr(item, "claim", None) or ""
        score = float(getattr(item, "similarity_score", getattr(item, "score", 0.0)) or 0.0)
        supporting = bool(getattr(item, "is_supporting", True))
        return str(snippet), str(claim), score, supporting

    def verify_evidence_date_mismatch(
        self,
        text: str,
        evidence_items: Optional[List[Any]] = None,
    ) -> Optional[float]:
        """Detect a date mismatch only when the complete evidence set lacks the claim year.

        A single secondary snippet containing another historical year is insufficient.
        The evidence must be relevant, sufficiently similar, and consistently conflict
        with the claimed year. Relational temporal statements are delegated to the
        event-relation resolver rather than treated as simple year mismatches.
        """
        if not evidence_items:
            return None
        text_lower = (text or "").lower()
        if self.RELATIVE_MARKERS.search(text_lower):
            return None
        claim_years = [int(y) for y in self.YEAR_PATTERN.findall(text)]
        if not claim_years:
            return None

        parsed = [self._get_evidence_fields(item) for item in evidence_items]
        all_evidence_years = [
            year for snippet, _, _, _ in parsed for year in [int(y) for y in self.YEAR_PATTERN.findall(snippet)]
        ]

        # Global support gate: if any evidence contains the claimed year, do not flag
        # a mismatch because another snippet may simply be contextual/background material.
        if any(year in all_evidence_years for year in claim_years):
            return None

        conflict_candidates = []
        for snippet, evidence_claim, similarity, supporting in parsed:
            if not snippet:
                continue
            ev_years = [int(y) for y in self.YEAR_PATTERN.findall(snippet)]
            if not ev_years:
                continue
            relevance = self._evidence_relevance(text, snippet, evidence_claim)
            if relevance < 2 and similarity < 0.75:
                continue
            for claim_year in claim_years:
                for evidence_year in ev_years:
                    if claim_year <= self.CURRENT_YEAR and evidence_year <= self.CURRENT_YEAR:
                        distance = abs(claim_year - evidence_year)
                        if distance >= 3:
                            conflict_candidates.append((distance, similarity, relevance, supporting))

        if not conflict_candidates:
            return None

        strongest = max(conflict_candidates, key=lambda x: (x[2], x[1], x[0]))
        _, similarity, relevance, supporting = strongest
        # Prefer explicit contradicting evidence; otherwise require stronger retrieval relevance.
        if not supporting or similarity >= 0.82 or relevance >= 3:
            return 0.90
        return None

    def _resolve_event_anchor_relation(
        self,
        text: str,
        evidence_items: Optional[List[Any]],
    ) -> Optional[Tuple[float, str]]:
        """Resolve no-year temporal relations from retrieved evidence anchors.

        This is intentionally knowledge-source agnostic: dates are obtained only from
        evidence already retrieved by Pillar 1. No entity/date pairs are hard-coded.
        """
        if not evidence_items or self.YEAR_PATTERN.search(text):
            return None
        relation = None
        match = None
        for name, pattern in self.RELATION_PATTERNS:
            candidate = pattern.search(text.lower())
            if candidate:
                relation, match = name, candidate
                break
        if not relation or match is None:
            return None

        left = text[: match.start()]
        right = text[match.end() :]
        if len(self._content_tokens(left)) < 1 or len(self._content_tokens(right)) < 1:
            return None

        def anchors(fragment: str) -> List[Tuple[int, float, int]]:
            result = []
            for item in evidence_items:
                snippet, evidence_claim, similarity, _ = self._get_evidence_fields(item)
                if not snippet:
                    continue
                relevance = int(self._evidence_relevance(fragment, snippet, evidence_claim))
                if relevance < 1 and similarity < 0.75:
                    continue
                years = [int(y) for y in self.YEAR_PATTERN.findall(snippet)]
                for year in years:
                    result.append((year, similarity, relevance))
            return result

        left_anchors = anchors(left)
        right_anchors = anchors(right)
        if not left_anchors or not right_anchors:
            return None

        left_years = [x[0] for x in left_anchors]
        right_years = [x[0] for x in right_anchors]
        confidence = max(min(1.0, (max(x[1] for x in left_anchors) + max(x[1] for x in right_anchors)) / 2), 0.70)

        contradiction = False
        if relation == "before":
            contradiction = min(left_years) >= max(right_years)
        elif relation == "after":
            contradiction = max(left_years) <= min(right_years)
        elif relation == "during":
            # Treat a single year anchor as a point and multiple years as an interval.
            left_point = min(left_years) if len(left_years) == 1 else (min(left_years) + max(left_years)) / 2
            right_start, right_end = min(right_years), max(right_years)
            contradiction = left_point < right_start or left_point > right_end

        if contradiction:
            return 0.92, f"Evidence-backed event-anchor contradiction: {relation} relation is incompatible with retrieved temporal spans."
        return 0.0, "Evidence-backed event-anchor relation is temporally compatible."

    def analyze_claim(
        self,
        text: str,
        query: Optional[str] = None,
        evaluation_year: Optional[int] = None,
        evidence_items: Optional[List[Any]] = None,
    ) -> TemporalAnalysisResult:
        eval_year = evaluation_year or self.CURRENT_YEAR
        q_str = query or ""
        years_found = [int(y) for y in self.YEAR_PATTERN.findall(text)]
        if not years_found and q_str:
            years_found = [int(y) for y in self.YEAR_PATTERN.findall(q_str)]
        has_temporal = bool(years_found)

        query_modality = self.detect_query_modality(q_str)
        response_modality = self.detect_response_modality(text, claim_text=text)
        modality = response_modality

        protected_modalities = {
            EpistemicModality.PREDICTION,
            EpistemicModality.HYPOTHETICAL,
            EpistemicModality.COUNTERFACTUAL,
            EpistemicModality.CONDITIONAL,
            EpistemicModality.NEGATED_FACT,
            EpistemicModality.FICTIONAL,
            EpistemicModality.QUOTED_CLAIM,
        }
        if modality in protected_modalities:
            status_map = {
                EpistemicModality.PREDICTION: TemporalStatus.FUTURE_PREDICTION,
                EpistemicModality.HYPOTHETICAL: TemporalStatus.HYPOTHETICAL,
                EpistemicModality.COUNTERFACTUAL: TemporalStatus.COUNTERFACTUAL,
                EpistemicModality.CONDITIONAL: TemporalStatus.CONDITIONAL,
                EpistemicModality.NEGATED_FACT: TemporalStatus.NEGATED_FACT,
                EpistemicModality.FICTIONAL: TemporalStatus.FICTIONAL,
                EpistemicModality.QUOTED_CLAIM: TemporalStatus.UNKNOWN,
            }
            return TemporalAnalysisResult(
                has_temporal_expression=has_temporal,
                extracted_years=years_found,
                modality=modality,
                temporal_status=status_map[modality],
                temporal_inconsistency_score=0.0,
                protected_from_temporal_penalty=True,
                reasoning=(
                    f"Response-level epistemic modality detected ({modality.value}); "
                    f"query modality ({query_modality.value}) was evaluated independently."
                ),
                query_modality=query_modality,
                response_modality=response_modality,
            )

        future_years = [year for year in years_found if year > eval_year]
        text_lower = text.lower()
        if future_years and any(verb in text_lower for verb in self.PAST_ACTION_VERBS + ["capital", "president", "winner"]):
            return TemporalAnalysisResult(
                True, years_found, EpistemicModality.FUTURE_FACT_ASSERTION,
                TemporalStatus.FUTURE_IMPOSSIBLE_FACT, 0.92, False,
                f"Asserted completed fact with future event year ({future_years[0]} > {eval_year}).",
                query_modality, response_modality,
            )

        mismatch = self.verify_evidence_date_mismatch(text, evidence_items)
        if mismatch is not None:
            return TemporalAnalysisResult(
                True, years_found, EpistemicModality.ASSERTED_FACT,
                TemporalStatus.DATE_MISMATCH, mismatch, False,
                "Global evidence-set date mismatch detected for the asserted claim.",
                query_modality, response_modality,
            )

        anchor_result = self._resolve_event_anchor_relation(text, evidence_items)
        if anchor_result is not None and anchor_result[0] > 0.0:
            return TemporalAnalysisResult(
                True, years_found, EpistemicModality.ASSERTED_FACT,
                TemporalStatus.DATE_MISMATCH, anchor_result[0], False,
                anchor_result[1], query_modality, response_modality,
            )

        if self.RELATIVE_MARKERS.search(text_lower):
            return TemporalAnalysisResult(
                bool(years_found) or bool(self.RELATIVE_MARKERS.search(text_lower)),
                years_found, EpistemicModality.ASSERTED_FACT,
                TemporalStatus.TIME_RELATIVE, 0.0, True,
                "Relational temporal operator detected; no naive year-distance penalty applied.",
                query_modality, response_modality,
            )

        if "between " in text_lower and " and " in text_lower and len(years_found) >= 2:
            return TemporalAnalysisResult(
                True, years_found, EpistemicModality.ASSERTED_FACT,
                TemporalStatus.DATE_RANGE, 0.0, True,
                f"Valid historical date range [{years_found[0]}-{years_found[1]}] statement.",
                query_modality, response_modality,
            )

        if has_temporal:
            return TemporalAnalysisResult(
                True, years_found, EpistemicModality.ASSERTED_FACT,
                TemporalStatus.PAST_FACT if max(years_found) <= eval_year else TemporalStatus.UNKNOWN,
                0.0, True, "Historical or present temporal expression within valid timeline bounds.",
                query_modality, response_modality,
            )

        return TemporalAnalysisResult(
            False, [], EpistemicModality.ASSERTED_FACT,
            TemporalStatus.PRESENT_STATE, 0.0, True,
            "No temporal expressions or resolvable event relations detected.",
            query_modality, response_modality,
        )
