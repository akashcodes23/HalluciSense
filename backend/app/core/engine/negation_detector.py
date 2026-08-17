"""Rule-Based Negation and Polarity Reversal Detector for HalluciSense Enhanced P1.

Detects explicit polarity mismatches, negation insertions/deletions, and antonymic predicate
flips between claim statements and retrieved evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Set


class Polarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


# Standard negation particles and auxiliary contractions
NEGATION_CUES: Set[str] = {
    "not", "never", "no", "none", "neither", "nor", "cannot", "can't", "cant",
    "won't", "wont", "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "weren't", "werent",
    "hasn't", "hasnt", "haven't", "havent", "hadn't", "hadnt", "without",
    "unable", "impossible", "fails", "failed", "failing", "lacks", "lacking",
    "disproved", "disproven", "refutes", "refuted", "denies", "denied",
}

# Antonymic action / direction predicate pairs
OPPOSITE_PAIRS = [
    ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
    ("higher", "lower"), ("raise", "lower"), ("raises", "lowers"),
    ("accelerate", "decelerate"), ("accelerates", "decelerates"),
    ("synthesise", "degrade"), ("synthesises", "degrades"), ("synthesize", "degrade"),
    ("activate", "inhibit"), ("activates", "inhibits"), ("activated", "inhibited"),
    ("stimulate", "suppress"), ("stimulates", "suppresses"), ("stimulated", "suppressed"),
    ("promote", "suppress"), ("promotes", "suppresses"), ("promoted", "suppressed"),
    ("promote", "prevent"), ("promotes", "prevents"), ("promoted", "prevented"),
    ("agonist", "antagonist"), ("absorb", "emit"), ("absorbs", "emits"),
    ("attract", "repel"), ("attracts", "repels"), ("attraction", "repulsion"),
    ("expand", "contract"), ("expands", "contracts"), ("expansion", "contraction"),
    ("producer", "consumer"), ("producers", "consumers"),
    ("exothermic", "endothermic"), ("aerobic", "anaerobic"),
    ("dominant", "recessive"), ("converge", "diverge"), ("converges", "diverges"),
    ("inflammatory", "anti-inflammatory"),
]


@dataclass
class PolarityAnalysis:
    claim_polarity: Polarity
    evidence_polarity: Polarity
    negation_inversion_detected: bool
    antonym_inversion_detected: bool
    detected_cues: List[str]
    confidence_penalty: float
    explanation: str


class NegationDetector:
    """Detects polarity inversion between claim text and retrieved evidence."""

    def __init__(self):
        self.negation_pattern = re.compile(
            r"\b(" + "|".join(re.escape(c) for c in sorted(NEGATION_CUES, key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        )

    def extract_negation_cues(self, text: str) -> List[str]:
        """Find all negation cues present in text."""
        return [m.group(0).lower() for m in self.negation_pattern.finditer(text)]

    def get_polarity(self, text: str) -> Tuple[Polarity, List[str]]:
        """Determine broad sentence polarity and extracted cues."""
        cues = self.extract_negation_cues(text)
        # An odd number of negative cues typically indicates negative polarity
        if len(cues) % 2 == 1:
            return Polarity.NEGATIVE, cues
        return Polarity.POSITIVE, cues

    def check_antonym_conflict(self, claim: str, evidence: str) -> Optional[Tuple[str, str]]:
        """Check if claim uses one pole of a predicate pair while evidence uses the opposite."""
        c_lower = claim.lower()
        e_lower = evidence.lower()
        c_words = set(re.findall(r"\b[\w-]+\b", c_lower))
        e_words = set(re.findall(r"\b[\w-]+\b", e_lower))

        for w1, w2 in OPPOSITE_PAIRS:
            if (w1 in c_words and w2 in e_words) or (w2 in c_words and w1 in e_words):
                return (w1, w2)
            if (w1 in c_lower and w2 in e_lower) or (w2 in c_lower and w1 in e_lower):
                return (w1, w2)
        return None

    def analyze(self, claim: str, evidence: str) -> PolarityAnalysis:
        """
        Compare polarity and predicate alignment between claim and evidence.
        """
        c_pol, c_cues = self.get_polarity(claim)
        e_pol, e_cues = self.get_polarity(evidence)

        # Check for direct negation mismatch (e.g. claim says 'Mitochondria do not contain DNA' vs evidence 'contains DNA')
        neg_mismatch = (c_pol != e_pol) and (len(c_cues) > 0 or len(e_cues) > 0)

        # Check for antonym clash (e.g. 'inhibits' vs 'activates')
        antonym_match = self.check_antonym_conflict(claim, evidence)

        if neg_mismatch and len(c_cues) > 0:
            return PolarityAnalysis(
                claim_polarity=c_pol,
                evidence_polarity=e_pol,
                negation_inversion_detected=True,
                antonym_inversion_detected=bool(antonym_match),
                detected_cues=c_cues,
                confidence_penalty=0.85,
                explanation=f"Polarity reversal: claim contains negation '{', '.join(c_cues)}' while evidence affirms the predicate.",
            )

        if antonym_match:
            w1, w2 = antonym_match
            return PolarityAnalysis(
                claim_polarity=c_pol,
                evidence_polarity=e_pol,
                negation_inversion_detected=False,
                antonym_inversion_detected=True,
                detected_cues=[],
                confidence_penalty=0.80,
                explanation=f"Antonym predicate conflict: claim uses '{w1}' whereas evidence describes '{w2}'.",
            )

        return PolarityAnalysis(
            claim_polarity=c_pol,
            evidence_polarity=e_pol,
            negation_inversion_detected=False,
            antonym_inversion_detected=False,
            detected_cues=[],
            confidence_penalty=0.0,
            explanation="No polarity inversion detected.",
        )
