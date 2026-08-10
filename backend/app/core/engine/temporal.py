"""Temporal Claim Analysis Engine for HalluciSense (Phase 4 Context-Aware Modality Resolution).

Extracts temporal expressions, performs claim-level context-aware epistemic modality resolution
(Asserted Fact, Prediction, Hypothetical, Counterfactual, Conditional, Negated Fact, Fictional, Quoted),
and evaluates temporal compatibility between event dates and evaluation time context.

Integrates with Pillar 1 (Factual Verification) to detect ungrounded future factual assertions
and historical date mismatches while protecting predictions, hypotheticals, conditionals, negations, and fiction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple


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
    """Generalizable Context-Aware Temporal Claim Analysis Component."""

    CURRENT_YEAR: int = 2026

    # Regex for 4-digit years between 1000 and 2100
    YEAR_PATTERN = re.compile(r"\b(1\d{3}|20\d{2}|2100)\b")

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
        "were to have", "if x had"
    ]

    CONDITIONAL_MARKERS = [
        "if ", "if we", "if it", "were to", "had it been"
    ]

    FICTION_MARKERS = [
        "fictional", "in the novel", "in the movie", "in the sci-fi story",
        "in fiction", "comic book", "mythological", "fantasy world", "video game",
        "game", "in the video game", "sci-fi", "anime", "literary work", "cyberpunk"
    ]

    NEGATION_MARKERS = [
        "did not", "had not", "has not", "was not", "were not", "no evidence",
        "no human", "never", "claim that", "false that", "untrue that", "before 19",
        "before 20"
    ]

    PAST_ACTION_VERBS = [
        "won", "released", "discovered", "defeated", "built", "launched",
        "created", "invented", "died", "signed", "passed", "elected",
        "hosted", "occurred", "took place", "ended", "completed", "published", "landed"
    ]

    def detect_modality(
        self,
        query: str,
        text: str,
        claim_text: Optional[str] = None,
    ) -> EpistemicModality:
        """Identify claim-level context-aware epistemic modality across query, response, and claim text."""
        combined_context = f"{query} {text}".lower()
        claim_lower = (claim_text or text).lower().strip()
        query_lower = (query or "").lower().strip()

        # 1. Fictional / Sci-fi context check
        for marker in self.FICTION_MARKERS:
            if marker in combined_context or marker in claim_lower:
                return EpistemicModality.FICTIONAL

        # 2. Negation / Non-assertion check
        for marker in self.NEGATION_MARKERS:
            if marker in claim_lower:
                return EpistemicModality.NEGATED_FACT

        # 3. Counterfactual check
        for marker in self.COUNTERFACTUAL_MARKERS:
            if marker in combined_context or marker in claim_lower:
                return EpistemicModality.COUNTERFACTUAL

        # 4. Hypothetical check
        for marker in self.HYPOTHETICAL_MARKERS:
            if marker in combined_context or marker in claim_lower:
                return EpistemicModality.HYPOTHETICAL

        # 5. Conditional check
        if re.search(r"\bif\b", claim_lower) or re.search(r"\bif\b", query_lower):
            return EpistemicModality.CONDITIONAL

        for marker in self.CONDITIONAL_MARKERS:
            if marker in combined_context or marker in claim_lower:
                return EpistemicModality.CONDITIONAL

        # 6. Prediction check
        for marker in self.PREDICTION_MARKERS:
            if marker in combined_context or marker in claim_lower:
                return EpistemicModality.PREDICTION

        return EpistemicModality.ASSERTED_FACT

    def verify_evidence_date_mismatch(
        self,
        text: str,
        evidence_items: Optional[List[Any]] = None,
    ) -> Optional[float]:
        """Check for historical date mismatches between claim and retrieved evidence snippets."""
        if not evidence_items:
            return None

        text_lower = text.lower()
        # Protect date range statements (e.g. "between 1914 and 1918") from internal mismatch false alarm
        if "between " in text_lower and " and " in text_lower:
            return None

        claim_years = [int(y) for y in self.YEAR_PATTERN.findall(text)]
        if not claim_years:
            return None

        for item in evidence_items:
            if isinstance(item, dict):
                snippet = item.get("snippet", "") or item.get("claim", "")
            else:
                snippet = getattr(item, "snippet", None) or getattr(item, "claim", None) or str(item)
            ev_years = [int(y) for y in self.YEAR_PATTERN.findall(snippet)]
            if not ev_years:
                continue

            for cy in claim_years:
                # If claim year is directly present in the snippet, it's not a date mismatch
                if cy in ev_years:
                    continue

                if cy <= self.CURRENT_YEAR:
                    for ey in ev_years:
                        if ey <= self.CURRENT_YEAR and abs(cy - ey) >= 3:
                            claim_words = set(re.findall(r"\b[A-Za-z]{4,}\b", text.lower()))
                            snippet_words = set(re.findall(r"\b[A-Za-z]{4,}\b", snippet.lower()))
                            common = claim_words.intersection(snippet_words)
                            filtered_common = [w for w in common if w not in {"first", "second", "states", "united", "world", "national", "american", "year", "occurred", "prior"}]
                            if len(filtered_common) >= 1:
                                return 0.90
        return None

    def analyze_claim(
        self,
        text: str,
        query: Optional[str] = None,
        evaluation_year: Optional[int] = None,
        evidence_items: Optional[List[Any]] = None,
    ) -> TemporalAnalysisResult:
        """Analyze temporal expressions, context-aware modality, and temporal consistency."""
        eval_year = evaluation_year or self.CURRENT_YEAR
        q_str = query or ""

        # 1. Extract years
        years_found = [int(y) for y in self.YEAR_PATTERN.findall(text)]
        if not years_found and q_str:
            years_found = [int(y) for y in self.YEAR_PATTERN.findall(q_str)]

        has_temporal = len(years_found) > 0

        # 2. Detect Context-Aware Modality
        modality = self.detect_modality(q_str, text, claim_text=text)

        # Protected Non-Assertion Modalities (Predictions, Hypotheticals, Conditionals, Counterfactuals, Negations, Fiction)
        if modality in (
            EpistemicModality.PREDICTION,
            EpistemicModality.HYPOTHETICAL,
            EpistemicModality.COUNTERFACTUAL,
            EpistemicModality.CONDITIONAL,
            EpistemicModality.NEGATED_FACT,
            EpistemicModality.FICTIONAL,
            EpistemicModality.QUOTED_CLAIM,
        ):
            status_map = {
                EpistemicModality.PREDICTION: TemporalStatus.FUTURE_PREDICTION,
                EpistemicModality.HYPOTHETICAL: TemporalStatus.HYPOTHETICAL,
                EpistemicModality.COUNTERFACTUAL: TemporalStatus.COUNTERFACTUAL,
                EpistemicModality.CONDITIONAL: TemporalStatus.CONDITIONAL,
                EpistemicModality.NEGATED_FACT: TemporalStatus.NEGATED_FACT,
                EpistemicModality.FICTIONAL: TemporalStatus.FICTIONAL,
                EpistemicModality.QUOTED_CLAIM: TemporalStatus.UNKNOWN,
            }
            status = status_map.get(modality, TemporalStatus.UNKNOWN)
            return TemporalAnalysisResult(
                has_temporal_expression=has_temporal,
                extracted_years=years_found,
                modality=modality,
                temporal_status=status,
                temporal_inconsistency_score=0.0,
                protected_from_temporal_penalty=True,
                reasoning=f"Protected epistemic modality detected ({modality.value}). Zero temporal penalty applied.",
            )

        # 3. Analyze Asserted Facts with Future Years
        future_years = [y for y in years_found if y > eval_year]
        text_lower = text.lower()

        # If future year is asserted as a completed fact
        if future_years and any(verb in text_lower for verb in self.PAST_ACTION_VERBS + ["capital", "president", "winner"]):
            return TemporalAnalysisResult(
                has_temporal_expression=True,
                extracted_years=years_found,
                modality=EpistemicModality.FUTURE_FACT_ASSERTION,
                temporal_status=TemporalStatus.FUTURE_IMPOSSIBLE_FACT,
                temporal_inconsistency_score=0.92,
                protected_from_temporal_penalty=False,
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
                protected_from_temporal_penalty=False,
                reasoning="Historical date mismatch detected between claim and retrieved ground-truth evidence.",
            )

        # 5. Date Range check
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

        # 6. Standard Past / Present Fact
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
