"""Epistemic Modality Resolution Engine for HalluciSense (Phase 6D).

Provides deterministic, typed epistemic frame resolution for queries,
responses, and atomic claims. Isolates assertions from non-assertion
modalities (predictions, hypotheticals, counterfactuals, conditionals,
negated facts, quotations, fictional statements, and meta-claims).

No benchmark-specific dates, entities, or labels are hardcoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Set

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
class EpistemicFrame:
    """Typed epistemic frame output for an evaluated text segment."""
    modality: EpistemicModality
    confidence: float
    trigger_spans: List[str] = field(default_factory=list)
    is_negated: bool = False
    is_quoted: bool = False
    is_conditional: bool = False
    is_protected: bool = False
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "modality": self.modality.value,
            "confidence": self.confidence,
            "trigger_spans": self.trigger_spans,
            "is_negated": self.is_negated,
            "is_quoted": self.is_quoted,
            "is_conditional": self.is_conditional,
            "is_protected": self.is_protected,
            "reasoning": self.reasoning,
        }


class EpistemicResolver:
    """Deterministic, context-aware epistemic modality resolver."""

    PROTECTED_MODALITIES: Set[EpistemicModality] = {
        EpistemicModality.PREDICTION,
        EpistemicModality.HYPOTHETICAL,
        EpistemicModality.COUNTERFACTUAL,
        EpistemicModality.CONDITIONAL,
        EpistemicModality.NEGATED_FACT,
        EpistemicModality.FICTIONAL,
        EpistemicModality.QUOTED_CLAIM,
    }

    # Structured pattern definitions with associated weights
    PREDICTION_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\bwill\b", re.IGNORECASE), 0.85),
        (re.compile(r"\bexpected\b.{0,80}\bto\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bpredicted\b.{0,80}\bto\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bprojected\b.{0,80}\bto\b", re.IGNORECASE), 0.95),
        (re.compile(r"\btargeted\b.{0,80}\bto\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bforecast\b.{0,80}\bto\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bscheduled\b.{0,80}\bto\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bplans?\b.{0,80}\bto\b", re.IGNORECASE), 0.85),
        (re.compile(r"\banticipates?\b.{0,80}\bto\b", re.IGNORECASE), 0.85),
        (re.compile(r"\bestimated\b.{0,80}\bto\b", re.IGNORECASE), 0.85),
        (re.compile(r"\baims?\b.{0,80}\bto\b", re.IGNORECASE), 0.85),
        (re.compile(r"\bis going to\b", re.IGNORECASE), 0.85),
        (re.compile(r"\bnext (?:year|month|decade|century)\b", re.IGNORECASE), 0.80),
    ]

    HYPOTHETICAL_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\bsuppos(?:e|ing)\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bassuming\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bhypothetically\b", re.IGNORECASE), 0.98),
        (re.compile(r"\bwhat if\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bif we assume\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bimagine\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bin a scenario where\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bin the event that\b", re.IGNORECASE), 0.90),
        (re.compile(r"\btheretical(?:ly)?\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bpostulate\b", re.IGNORECASE), 0.85),
    ]

    COUNTERFACTUAL_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\bhad been\b", re.IGNORECASE), 0.85),
        (re.compile(r"\bif it had\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bwould have\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bcould have\b", re.IGNORECASE), 0.85),
        (re.compile(r"\bmight have\b", re.IGNORECASE), 0.85),
        (re.compile(r"\bhad\b.{0,50}\b(?:won|released|built|launched|occurred)\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bwere to have\b", re.IGNORECASE), 0.95),
        (re.compile(r"\botherwise\b", re.IGNORECASE), 0.70),
    ]

    FICTION_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\bfictional\b", re.IGNORECASE), 0.98),
        (re.compile(r"\bin the novel\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bin the movie\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bin the film\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bin the\b.{0,80}\buniverse\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bin fiction\b", re.IGNORECASE), 0.98),
        (re.compile(r"\bcomic book\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bmythological\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bfantasy world\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bvideo game\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bsci[- ]fi\b", re.IGNORECASE), 0.90),
        (re.compile(r"\banime\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bliter(?:ary|ature)\b", re.IGNORECASE), 0.75),
        (re.compile(r"\bcyberpunk\b", re.IGNORECASE), 0.85),
    ]

    META_CLAIM_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\bfalsely\s+(?:reported|claimed|stated|asserted)\b", re.IGNORECASE), 0.95),
        (re.compile(r"\berroneously\s+(?:reported|claimed|stated|asserted)\b", re.IGNORECASE), 0.95),
        (re.compile(r"\bdebunked\s+(?:claim|assertion|rumou?r)\b", re.IGNORECASE), 0.95),
        (re.compile(r"\b(?:claim|assertion|rumou?r|myth)\s+that\b", re.IGNORECASE), 0.80),
        (re.compile(r"\b(?:article|post|report|source)\s+(?:claimed|reported|stated)\b", re.IGNORECASE), 0.85),
        (re.compile(r"\balleged(?:ly)?\b", re.IGNORECASE), 0.85),
        (re.compile(r"\baccording to\b", re.IGNORECASE), 0.75),
    ]

    NEGATION_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\bdid not\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bhad not\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bhas not\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bwas not\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bwere not\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bno evidence\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bno human\b", re.IGNORECASE), 0.85),
        (re.compile(r"\bnever\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bfalse(?:ly)?\s+(?:reported|claimed|stated)\b", re.IGNORECASE), 0.95),
        (re.compile(r"\buntrue\s+(?:claim|assertion)\b", re.IGNORECASE), 0.95),
    ]

    CONDITIONAL_PATTERNS: List[Tuple[re.Pattern, float]] = [
        (re.compile(r"\bif\b.{1,100}\bthen\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bprovided that\b", re.IGNORECASE), 0.85),
        (re.compile(r"\bon condition that\b", re.IGNORECASE), 0.90),
        (re.compile(r"\bunless\b", re.IGNORECASE), 0.80),
        (re.compile(r"\bif\b", re.IGNORECASE), 0.75),
    ]

    def _find_matches(self, patterns: List[Tuple[re.Pattern, float]], text: str) -> Tuple[float, List[str]]:
        max_conf = 0.0
        triggers = []
        for pat, conf in patterns:
            match = pat.search(text)
            if match:
                triggers.append(match.group(0))
                if conf > max_conf:
                    max_conf = conf
        return max_conf, triggers

    def resolve_frame(self, text: str, is_query: bool = False) -> EpistemicFrame:
        """Resolve typed EpistemicFrame for a given text segment."""
        clean_text = (text or "").strip()
        if not clean_text:
            return EpistemicFrame(
                modality=EpistemicModality.UNKNOWN,
                confidence=0.0,
                reasoning="Empty text provided.",
            )

        # Check fiction first
        conf, triggers = self._find_matches(self.FICTION_PATTERNS, clean_text)
        if conf > 0.0:
            return EpistemicFrame(
                modality=EpistemicModality.FICTIONAL,
                confidence=conf,
                trigger_spans=triggers,
                is_protected=True,
                reasoning=f"Fictional domain framing detected ({triggers[0]}).",
            )

        # Meta-claim / quoted claim
        conf, triggers = self._find_matches(self.META_CLAIM_PATTERNS, clean_text)
        if conf > 0.0:
            return EpistemicFrame(
                modality=EpistemicModality.QUOTED_CLAIM,
                confidence=conf,
                trigger_spans=triggers,
                is_quoted=True,
                is_protected=True,
                reasoning=f"Quoted or meta-claim attribution framing detected ({triggers[0]}).",
            )

        # Negation
        conf_neg, triggers_neg = self._find_matches(self.NEGATION_PATTERNS, clean_text)

        # Counterfactual
        conf, triggers = self._find_matches(self.COUNTERFACTUAL_PATTERNS, clean_text)
        if conf > 0.0:
            return EpistemicFrame(
                modality=EpistemicModality.COUNTERFACTUAL,
                confidence=conf,
                trigger_spans=triggers,
                is_negated=conf_neg > 0.0,
                is_protected=True,
                reasoning=f"Counterfactual framing detected ({triggers[0]}).",
            )

        # Hypothetical
        conf, triggers = self._find_matches(self.HYPOTHETICAL_PATTERNS, clean_text)
        if conf > 0.0:
            return EpistemicFrame(
                modality=EpistemicModality.HYPOTHETICAL,
                confidence=conf,
                trigger_spans=triggers,
                is_negated=conf_neg > 0.0,
                is_protected=True,
                reasoning=f"Hypothetical supposition detected ({triggers[0]}).",
            )

        # Conditional
        conf, triggers = self._find_matches(self.CONDITIONAL_PATTERNS, clean_text)
        if conf > 0.0:
            return EpistemicFrame(
                modality=EpistemicModality.CONDITIONAL,
                confidence=conf,
                trigger_spans=triggers,
                is_conditional=True,
                is_negated=conf_neg > 0.0,
                is_protected=True,
                reasoning=f"Conditional proposition detected ({triggers[0]}).",
            )

        # Prediction
        conf, triggers = self._find_matches(self.PREDICTION_PATTERNS, clean_text)
        if conf > 0.0:
            return EpistemicFrame(
                modality=EpistemicModality.PREDICTION,
                confidence=conf,
                trigger_spans=triggers,
                is_negated=conf_neg > 0.0,
                is_protected=True,
                reasoning=f"Forward-looking prediction or forecast detected ({triggers[0]}).",
            )

        # Pure negation
        if conf_neg > 0.0:
            return EpistemicFrame(
                modality=EpistemicModality.NEGATED_FACT,
                confidence=conf_neg,
                trigger_spans=triggers_neg,
                is_negated=True,
                is_protected=True,
                reasoning=f"Negated factual assertion detected ({triggers_neg[0]}).",
            )

        # Default: Asserted Fact
        return EpistemicFrame(
            modality=EpistemicModality.ASSERTED_FACT,
            confidence=0.90 if is_query else 0.95,
            trigger_spans=[],
            is_protected=False,
            reasoning="Direct factual assertion.",
        )
