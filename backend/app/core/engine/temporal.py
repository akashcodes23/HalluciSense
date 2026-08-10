"""Temporal Claim Analysis Engine for HalluciSense (Phase 6).

Phase 6 hardens the Phase 4 temporal layer against blind-holdout failures:
- query and response modality are resolved independently;
- date mismatch checks operate over the complete evidence set rather than one
  snippet at a time;
- relational temporal language is protected from naive year-difference logic;
- prediction, quotation/meta-claim, and fictional markers are structural and
  tolerate intervening words.

No entity/date-specific benchmark rules are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


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


class TemporalClaimEngine:
    """Generalizable context-aware temporal claim analysis component."""

    CURRENT_YEAR: int = 2026
    YEAR_PATTERN = re.compile(r"\b(1\d{3}|20\d{2}|2100)\b")

    # Structural patterns intentionally avoid benchmark/entity-specific strings.
    PREDICTION_PATTERNS = [
        re.compile(r"\bwill\b"),
        re.compile(r"\bexpected\b.{0,80}\bto\b"),
        re.compile(r"\bpredicted\b.{0,80}\bto\b"),
        re.compile(r"\bprojected\b.{0,80}\bto\b"),
        re.compile(r"\btargeted\b.{0,80}\bto\b"),
        re.compile(r"\bforecast\b.{0,80}\bto\b"),
        re.compile(r"\bscheduled\b.{0,80}\bto\b"),
        re.compile(r"\bplans?\b.{0,80}\bto\b"),
        re.compile(r"\banticipates?\b.{0,80}\bto\b"),
        re.compile(r"\bestimated\b.{0,80}\bto\b"),
        re.compile(r"\baims?\b.{0,80}\bto\b"),
    ]

    HYPOTHETICAL_PATTERNS = [
        re.compile(r"\bsuppos(?:e|ing)\b"),
        re.compile(r"\bassuming\b"),
        re.compile(r"\bhypothetically\b"),
        re.compile(r"\bwhat if\b"),
        re.compile(r"\bif we assume\b"),
        re.compile(r"\bimagine\b"),
        re.compile(r"\bin a scenario where\b"),
        re.compile(r"\bin the event that\b"),
    ]

    COUNTERFACTUAL_PATTERNS = [
        re.compile(r"\bhad been\b"),
        re.compile(r"\bif it had\b"),
        re.compile(r"\bwould have been\b"),
        re.compile(r"\bhad\b.{0,50}\bwon\b"),
        re.compile(r"\bhad\b.{0,50}\breleased\b"),
        re.compile(r"\bwere to have\b"),
    ]

    FICTION_PATTERNS = [
        re.compile(r"\bfictional\b"),
        re.compile(r"\bin the novel\b"),
        re.compile(r"\bin the movie\b"),
        re.compile(r"\bin the\b.{0,80}\buniverse\b"),
        re.compile(r"\bin fiction\b"),
        re.compile(r"\bcomic book\b"),
        re.compile(r"\bmythological\b"),
        re.compile(r"\bfantasy world\b"),
        re.compile(r"\bvideo game\b"),
        re.compile(r"\bin the video game\b"),
        re.compile(r"\bsci[- ]fi\b"),
        re.compile(r"\banime\b"),
        re.compile(r"\bliter(?:ary|ature)\b"),
        re.compile(r"\bcyberpunk\b"),
    ]

    META_CLAIM_PATTERNS = [
        re.compile(r"\bfalsely\s+(?:reported|claimed|stated|asserted)\b"),
        re.compile(r"\berroneously\s+(?:reported|claimed|stated|asserted)\b"),
        re.compile(r"\bdebunked\s+(?:claim|assertion|rumou?r)\b"),
        re.compile(r"\b(?:claim|assertion|rumou?r|myth)\s+that\b"),
        re.compile(r"\b(?:article|post|report|source)\s+(?:claimed|reported|stated)\b"),
        re.compile(r"\balleged(?:ly)?\b"),
        re.compile(r"\baccording to\b"),
    ]

    NEGATION_PATTERNS = [
        re.compile(r"\bdid not\b"),
        re.compile(r"\bhad not\b"),
        re.compile(r"\bhas not\b"),
        re.compile(r"\bwas not\b"),
        re.compile(r"\bwere not\b"),
        re.compile(r"\bno evidence\b"),
        re.compile(r"\bno human\b"),
        re.compile(r"\bnever\b"),
        re.compile(r"\bfalse(?:ly)?\s+(?:reported|claimed|stated)\b"),
        re.compile(r"\buntrue\s+(?:claim|assertion)\b"),
    ]

    PAST_ACTION_VERBS = [
        "won", "released", "discovered", "defeated", "built", "launched",
        "created", "invented", "died", "signed", "passed", "elected",
        "hosted", "occurred", "took place", "ended", "completed", "published", "landed"
    ]

    RELATIONAL_PATTERNS = [
        re.compile(r"\b(before|after|since|prior to|following|preceding|preceded by|subsequent to)\b"),
        re.compile(r"\b(?:years?|decades?|centuries?)\s+(?:before|after)\b"),
        re.compile(r"\b(?:earlier|later)\s+than\b"),
    ]

    def _matches_any(self, patterns: List[re.Pattern], text: str) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    def detect_query_modality(self, query: str) -> EpistemicModality:
        """Resolve modality of the user query only; never use it to classify the response."""
        query_lower = (query or "").lower().strip()
        if not query_lower:
            return EpistemicModality.UNKNOWN
        if self._matches_any(self.FICTION_PATTERNS, query_lower):
            return EpistemicModality.FICTIONAL
        if self._matches_any(self.COUNTERFACTUAL_PATTERNS, query_lower):
            return EpistemicModality.COUNTERFACTUAL
        if self._matches_any(self.HYPOTHETICAL_PATTERNS, query_lower):
            return EpistemicModality.HYPOTHETICAL
        if re.search(r"\bif\b", query_lower):
            return EpistemicModality.CONDITIONAL
        if self._matches_any(self.PREDICTION_PATTERNS, query_lower):
            return EpistemicModality.PREDICTION
        return EpistemicModality.ASSERTED_FACT

    def detect_modality(
        self,
        query: str,
        text: str,
        claim_text: Optional[str] = None,
    ) -> EpistemicModality:
        """Resolve response/claim modality independently of query modality.

        ``query`` is retained for API compatibility and diagnostics, but its
        hypothetical/conditional markers cannot protect an asserted response.
        """
        del query
        claim_lower = (claim_text or text or "").lower().strip()

        if self._matches_any(self.FICTION_PATTERNS, claim_lower):
            return EpistemicModality.FICTIONAL
        if self._matches_any(self.META_CLAIM_PATTERNS, claim_lower):
            return EpistemicModality.QUOTED_CLAIM
        if self._matches_any(self.NEGATION_PATTERNS, claim_lower):
            return EpistemicModality.NEGATED_FACT
        if self._matches_any(self.COUNTERFACTUAL_PATTERNS, claim_lower):
            return EpistemicModality.COUNTERFACTUAL
        if self._matches_any(self.HYPOTHETICAL_PATTERNS, claim_lower):
            return EpistemicModality.HYPOTHETICAL
        if re.search(r"\bif\b", claim_lower):
            return EpistemicModality.CONDITIONAL
        if self._matches_any(self.PREDICTION_PATTERNS, claim_lower):
            return EpistemicModality.PREDICTION
        return EpistemicModality.ASSERTED_FACT

    def verify_evidence_date_mismatch(
        self,
        text: str,
        evidence_items: Optional[List[Any]] = None,
    ) -> Optional[float]:
        """Check claim/evidence dates using global evidence consistency.

        A claim year is considered supported when it appears in *any* retrieved
        evidence item. This prevents background years in one snippet from
        overriding a matching year in another relevant snippet.
        """
        if not evidence_items:
            return None

        text_lower = text.lower()
        if "between " in text_lower and " and " in text_lower:
            return None

        claim_years = [int(y) for y in self.YEAR_PATTERN.findall(text)]
        if not claim_years:
            return None

        # Comparative language requires event/event reasoning rather than
        # treating the referenced year as the event's own date.
        if self._matches_any(self.RELATIONAL_PATTERNS, text_lower):
            return None

        evidence_years = set()
        evidence_records = []
        for item in evidence_items:
            if isinstance(item, dict):
                snippet = item.get("snippet", "") or item.get("claim", "")
            else:
                snippet = getattr(item, "snippet", None) or getattr(item, "claim", None) or str(item)
            years = {int(y) for y in self.YEAR_PATTERN.findall(snippet)}
            if years:
                evidence_years.update(years)
                evidence_records.append((snippet, years))

        if not evidence_records:
            return None

        # Global support check: matching claim year anywhere in evidence wins.
        for claim_year in claim_years:
            if claim_year in evidence_years:
                continue

            if claim_year <= self.CURRENT_YEAR:
                for snippet, ev_years in evidence_records:
                    claim_words = set(re.findall(r"\b[A-Za-z]{4,}\b", text_lower))
                    snippet_words = set(re.findall(r"\b[A-Za-z]{4,}\b", snippet.lower()))
                    common = claim_words.intersection(snippet_words)
                    filtered_common = [
                        w for w in common
                        if w not in {
                            "first", "second", "states", "united", "world",
                            "national", "american", "year", "occurred", "prior"
                        }
                    ]
                    if any(abs(claim_year - ey) >= 3 for ey in ev_years) and len(filtered_common) >= 1:
                        return 0.90
        return None

    def analyze_claim(
        self,
        text: str,
        query: Optional[str] = None,
        evaluation_year: Optional[int] = None,
        evidence_items: Optional[List[Any]] = None,
    ) -> TemporalAnalysisResult:
        """Analyze temporal expressions and response-level epistemic modality."""
        eval_year = evaluation_year or self.CURRENT_YEAR
        q_str = query or ""
        years_found = [int(y) for y in self.YEAR_PATTERN.findall(text)]
        if not years_found and q_str:
            years_found = [int(y) for y in self.YEAR_PATTERN.findall(q_str)]

        has_temporal = bool(years_found)
        modality = self.detect_modality(q_str, text, claim_text=text)

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
                reasoning=f"Protected response modality detected ({modality.value}); temporal penalty suppressed without overriding NLI grounding.",
            )

        future_years = [y for y in years_found if y > eval_year]
        text_lower = text.lower()
        if future_years and any(verb in text_lower for verb in self.PAST_ACTION_VERBS + ["capital", "president", "winner"]):
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.FUTURE_FACT_ASSERTION,
                temporal_status=TemporalStatus.FUTURE_IMPOSSIBLE_FACT,
                temporal_inconsistency_score=0.92,
                protected_from_temporal_penalty=False,
                reasoning=f"Asserted completed fact with future event year ({future_years[0]} > {eval_year}).",
            )

        date_mismatch_score = self.verify_evidence_date_mismatch(text, evidence_items)
        if date_mismatch_score is not None and date_mismatch_score > 0.0:
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.DATE_MISMATCH,
                temporal_inconsistency_score=date_mismatch_score,
                protected_from_temporal_penalty=False,
                reasoning="Historical date mismatch detected after global evidence-year consistency check.",
            )

        if "between " in text_lower and " and " in text_lower and len(years_found) >= 2:
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.DATE_RANGE,
                temporal_inconsistency_score=0.0,
                protected_from_temporal_penalty=True,
                reasoning=f"Valid historical date range [{years_found[0]}-{years_found[1]}] statement.",
            )

        if self._matches_any(self.RELATIONAL_PATTERNS, text_lower):
            return TemporalAnalysisResult(
                has_temporal_expression=has_temporal,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.TIME_RELATIVE,
                temporal_inconsistency_score=0.0,
                protected_from_temporal_penalty=True,
                reasoning="Relational temporal language detected; requires event-to-event comparison rather than naive year mismatch.",
            )

        if has_temporal:
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.PAST_FACT if max(years_found) <= eval_year else TemporalStatus.UNKNOWN,
                temporal_inconsistency_score=0.0,
                protected_from_temporal_penalty=True,
                reasoning="Historical or present temporal expression within valid timeline bounds.",
            )

        return TemporalAnalysisResult(
            has_temporal_expression=False,
            extracted_years=[],
            modality=EpistemicModality.ASSERTED_FACT,
            temporal_status=TemporalStatus.PRESENT_STATE,
            temporal_inconsistency_score=0.0,
            protected_from_temporal_penalty=True,
            reasoning="No temporal expressions detected.",
        )
