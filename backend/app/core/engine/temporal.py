"""Temporal Claim Analysis Engine for HalluciSense.

Extracts temporal expressions, identifies epistemic modalities (Asserted Fact, Prediction,
Hypothetical, Counterfactual, Fiction), and evaluates temporal compatibility between event dates
and evaluation time context.

Integrates with Pillar 1 (Factual Verification) to detect ungrounded future factual assertions
and date mismatches while protecting legitimate predictions, hypotheticals, and fiction.
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
    FICTIONAL = "FICTIONAL"
    DATE_MISMATCH = "DATE_MISMATCH"
    TIME_RELATIVE = "TIME_RELATIVE"
    UNKNOWN = "UNKNOWN"


class EpistemicModality(str, Enum):
    ASSERTED_FACT = "ASSERTED_FACT"
    PREDICTION = "PREDICTION"
    HYPOTHETICAL = "HYPOTHETICAL"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    FICTION = "FICTION"


@dataclass
class TemporalAnalysisResult:
    has_temporal_expression: bool
    extracted_years: List[int]
    modality: EpistemicModality
    temporal_status: TemporalStatus
    temporal_inconsistency_score: float
    reasoning: str


class TemporalClaimEngine:
    """Generalizable Temporal Claim Analysis Component."""

    CURRENT_YEAR: int = 2026

    # Regex for 4-digit years between 1800 and 2100
    YEAR_PATTERN = re.compile(r"\b(1[89]\d\d|20\d\d|2100)\b")

    # Epistemic modality markers
    PREDICTION_MARKERS = [
        "will", "expected to", "predicted to", "projected to", "scheduled to",
        "is forecast to", "plans to", "anticipates", "estimated to", "aims to"
    ]

    HYPOTHETICAL_MARKERS = [
        "suppose", "assuming", "hypothetically", "what if", "if we assume",
        "imagine", "in a scenario where", "in the event that"
    ]

    COUNTERFACTUAL_MARKERS = [
        "had been", "if it had", "would have been", "had won", "had released",
        "were to have", "if X had"
    ]

    FICTION_MARKERS = [
        "fictional", "in the novel", "in the movie", "in the sci-fi story",
        "in fiction", "comic book", "mythological", "fantasy world"
    ]

    PAST_ACTION_VERBS = [
        "won", "released", "discovered", "defeated", "built", "launched",
        "created", "invented", "died", "signed", "passed", "elected",
        "hosted", "occurred", "took place", "ended", "completed", "published"
    ]

    def detect_modality(self, query: str, text: str) -> EpistemicModality:
        """Identify epistemic modality across query and response text."""
        combined = f"{query} {text}".lower()

        for marker in self.FICTION_MARKERS:
            if marker in combined:
                return EpistemicModality.FICTION

        for marker in self.COUNTERFACTUAL_MARKERS:
            if marker in combined:
                return EpistemicModality.COUNTERFACTUAL

        for marker in self.HYPOTHETICAL_MARKERS:
            if marker in combined:
                return EpistemicModality.HYPOTHETICAL

        for marker in self.PREDICTION_MARKERS:
            if marker in combined:
                return EpistemicModality.PREDICTION

        return EpistemicModality.ASSERTED_FACT

    def analyze_claim(
        self,
        text: str,
        query: Optional[str] = None,
        evaluation_year: Optional[int] = None,
    ) -> TemporalAnalysisResult:
        """Analyze temporal expressions, modality, and temporal consistency."""
        eval_year = evaluation_year or self.CURRENT_YEAR
        q_str = query or ""

        # 1. Extract years
        years_found = [int(y) for y in self.YEAR_PATTERN.findall(text)]
        if not years_found and q_str:
            years_found = [int(y) for y in self.YEAR_PATTERN.findall(q_str)]

        has_temporal = len(years_found) > 0

        # 2. Detect Modality
        modality = self.detect_modality(q_str, text)

        # If protected modality (Prediction, Hypothetical, Counterfactual, Fiction), no temporal inconsistency
        if modality in (
            EpistemicModality.PREDICTION,
            EpistemicModality.HYPOTHETICAL,
            EpistemicModality.COUNTERFACTUAL,
            EpistemicModality.FICTION,
        ):
            status_map = {
                EpistemicModality.PREDICTION: TemporalStatus.FUTURE_PREDICTION,
                EpistemicModality.HYPOTHETICAL: TemporalStatus.HYPOTHETICAL,
                EpistemicModality.COUNTERFACTUAL: TemporalStatus.COUNTERFACTUAL,
                EpistemicModality.FICTION: TemporalStatus.FICTIONAL,
            }
            status = status_map[modality]
            return TemporalAnalysisResult(
                has_temporal_expression=has_temporal,
                extracted_years=years_found,
                modality=modality,
                temporal_status=status,
                temporal_inconsistency_score=0.0,
                reasoning=f"Protected epistemic modality detected ({modality.value}). No temporal inconsistency assigned.",
            )

        # 3. Analyze Asserted Facts with Future Years or Temporal Contradictions
        future_years = [y for y in years_found if y > eval_year]

        text_lower = text.lower()
        has_past_verb = any(verb in text_lower for verb in self.PAST_ACTION_VERBS)

        if future_years and (has_past_verb or "won" in text_lower or "released" in text_lower or "discovered" in text_lower or "invented" in text_lower or "capital" in text_lower or "landed" in text_lower or "elected" in text_lower):
            # Asserted past action with a future date relative to evaluation year
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.FUTURE_IMPOSSIBLE_FACT,
                temporal_inconsistency_score=0.92,
                reasoning=f"Asserted completed fact with future event year ({future_years[0]} > {eval_year}). Temporal assertion is unverified / impossible.",
            )

        # 4. Analyze Historical Date Mismatch against Evidence
        date_mismatch_score = self.verify_evidence_date_mismatch(text, evidence_items)
        if date_mismatch_score is not None and date_mismatch_score > 0.0:
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.DATE_MISMATCH,
                temporal_inconsistency_score=date_mismatch_score,
                reasoning="Historical date mismatch detected between claim and retrieved ground-truth evidence.",
            )

        # 5. Date Range Contradiction check (e.g. "between 2014 and 2018, World War I took place")
        range_match = re.search(r"between\s+(1[89]\d\d|20\d\d)\s+and\s+(1[89]\d\d|20\d\d)", text_lower)
        if range_match:
            y1, y2 = int(range_match.group(1)), int(range_match.group(2))
            if date_mismatch_score is not None and date_mismatch_score > 0:
                return TemporalAnalysisResult(
                    has_temporal_expression=True,
                    extracted_years=[y1, y2],
                    modality=EpistemicModality.ASSERTED_FACT,
                    temporal_status=TemporalStatus.DATE_MISMATCH,
                    temporal_inconsistency_score=0.90,
                    reasoning=f"Contradictory date range [{y1}-{y2}] detected.",
                )

        # 6. Standard Past / Present Fact
        if has_temporal:
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.PAST_FACT if max(years_found) <= eval_year else TemporalStatus.UNKNOWN,
                temporal_inconsistency_score=0.0,
                reasoning="Historical or present temporal expression within valid timeline bounds.",
            )

        return TemporalAnalysisResult(
            has_temporal_expression=False,
            extracted_years=[],
            modality=EpistemicModality.ASSERTED_FACT,
            temporal_status=TemporalStatus.PRESENT_STATE,
            temporal_inconsistency_score=0.0,
            reasoning="No temporal expressions detected.",
        )

    def verify_evidence_date_mismatch(
        self,
        text: str,
        evidence_items: Optional[List[Any]] = None,
    ) -> Optional[float]:
        """Check for historical date mismatches between claim and retrieved evidence."""
        if not evidence_items:
            return None

        claim_years = [int(y) for y in self.YEAR_PATTERN.findall(text)]
        if not claim_years:
            return None

        for item in evidence_items:
            snippet = getattr(item, "snippet", "") if hasattr(item, "snippet") else str(item)
            ev_years = [int(y) for y in self.YEAR_PATTERN.findall(snippet)]
            if not ev_years:
                continue

            for cy in claim_years:
                # If claim year is directly supported in snippet, it's not a mismatch
                if cy in ev_years:
                    continue

                if cy <= self.CURRENT_YEAR:
                    for ey in ev_years:
                        if ey <= self.CURRENT_YEAR and abs(cy - ey) >= 3:
                            claim_words = set(re.findall(r"\b[A-Za-z]{4,}\b", text.lower()))
                            snippet_words = set(re.findall(r"\b[A-Za-z]{4,}\b", snippet.lower()))
                            common = claim_words.intersection(snippet_words)
                            filtered_common = [w for w in common if w not in {"first", "second", "states", "united", "world", "national", "american", "year", "occurred", "prior"}]
                            if len(filtered_common) >= 2:
                                return 0.90
        return None

    def analyze_claim(
        self,
        text: str,
        query: Optional[str] = None,
        evaluation_year: Optional[int] = None,
        evidence_items: Optional[List[Any]] = None,
    ) -> TemporalAnalysisResult:
        """Analyze temporal expressions, modality, and temporal consistency."""
        eval_year = evaluation_year or self.CURRENT_YEAR
        q_str = query or ""

        # 1. Extract years
        years_found = [int(y) for y in self.YEAR_PATTERN.findall(text)]
        if not years_found and q_str:
            years_found = [int(y) for y in self.YEAR_PATTERN.findall(q_str)]

        has_temporal = len(years_found) > 0

        # 2. Detect Modality
        modality = self.detect_modality(q_str, text)

        # If protected modality (Prediction, Hypothetical, Counterfactual, Fiction), no temporal inconsistency
        if modality in (
            EpistemicModality.PREDICTION,
            EpistemicModality.HYPOTHETICAL,
            EpistemicModality.COUNTERFACTUAL,
            EpistemicModality.FICTION,
        ):
            status_map = {
                EpistemicModality.PREDICTION: TemporalStatus.FUTURE_PREDICTION,
                EpistemicModality.HYPOTHETICAL: TemporalStatus.HYPOTHETICAL,
                EpistemicModality.COUNTERFACTUAL: TemporalStatus.COUNTERFACTUAL,
                EpistemicModality.FICTION: TemporalStatus.FICTIONAL,
            }
            status = status_map[modality]
            return TemporalAnalysisResult(
                has_temporal_expression=has_temporal,
                extracted_years=years_found,
                modality=modality,
                temporal_status=status,
                temporal_inconsistency_score=0.0,
                reasoning=f"Protected epistemic modality detected ({modality.value}). No temporal inconsistency assigned.",
            )

        # 3. Analyze Asserted Facts with Future Years or Temporal Contradictions
        future_years = [y for y in years_found if y > eval_year]

        text_lower = text.lower()
        has_past_verb = any(verb in text_lower for verb in self.PAST_ACTION_VERBS)

        if future_years and (has_past_verb or "won" in text_lower or "released" in text_lower or "discovered" in text_lower or "invented" in text_lower or "capital" in text_lower or "landed" in text_lower or "elected" in text_lower):
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.FUTURE_IMPOSSIBLE_FACT,
                temporal_inconsistency_score=0.92,
                reasoning=f"Asserted completed fact with future event year ({future_years[0]} > {eval_year}). Temporal assertion is unverified / impossible.",
            )

        # 4. Analyze Historical Date Mismatch against Evidence
        date_mismatch_score = self.verify_evidence_date_mismatch(text, evidence_items)
        if date_mismatch_score is not None and date_mismatch_score > 0.0:
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.DATE_MISMATCH,
                temporal_inconsistency_score=date_mismatch_score,
                reasoning="Historical date mismatch detected between claim and retrieved ground-truth evidence.",
            )

        # 5. Date Range Contradiction check
        range_match = re.search(r"between\s+(1[89]\d\d|20\d\d)\s+and\s+(1[89]\d\d|20\d\d)", text_lower)
        if range_match:
            y1, y2 = int(range_match.group(1)), int(range_match.group(2))
            if date_mismatch_score is not None and date_mismatch_score > 0:
                return TemporalAnalysisResult(
                    has_temporal_expression=True,
                    extracted_years=[y1, y2],
                    modality=EpistemicModality.ASSERTED_FACT,
                    temporal_status=TemporalStatus.DATE_MISMATCH,
                    temporal_inconsistency_score=0.90,
                    reasoning=f"Contradictory date range [{y1}-{y2}] detected.",
                )

        # 6. Standard Past / Present Fact
        if has_temporal:
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.ASSERTED_FACT,
                temporal_status=TemporalStatus.PAST_FACT if max(years_found) <= eval_year else TemporalStatus.UNKNOWN,
                temporal_inconsistency_score=0.0,
                reasoning="Historical or present temporal expression within valid timeline bounds.",
            )

        return TemporalAnalysisResult(
            has_temporal_expression=False,
            extracted_years=[],
            modality=EpistemicModality.ASSERTED_FACT,
            temporal_status=TemporalStatus.PRESENT_STATE,
            temporal_inconsistency_score=0.0,
            reasoning="No temporal expressions detected.",
        )
