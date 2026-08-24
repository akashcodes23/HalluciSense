"""Causal Direction Asymmetry Checker for HalluciSense Enhanced P1.

Identifies cause-effect relationships and detects directional inversions (A causes B vs B causes A)
between claims and reference evidence without external LLM dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Forward causal connectors: [CAUSE] -> [EFFECT]
FORWARD_CAUSAL_PATTERNS = [
    r"(.+?)\s+(?:always causes|always caused|causes|caused|causing)\s+(.+)",
    r"(.+?)\s+(?:leads to|led to|leading to)\s+(.+)",
    r"(.+?)\s+(?:results in|resulted in|resulting in)\s+(.+)",
    r"(.+?)\s+(?:produces|produced|producing)\s+(.+)",
    r"(.+?)\s+(?:induces|induced|inducing)\s+(.+)",
    r"(.+?)\s+(?:triggers|triggered|triggering)\s+(.+)",
    r"(.+?)\s+(?:drives|driven by|driving)\s+(.+)",
    r"(.+?)\s+(?:is(?:,?\s*however,?\s*|\s+\w+)?\s+a\s+(?:major\s+|primary\s+|known\s+|key\s+)?risk\s+factor\s+(?:for|in))\s+(.+)",
    r"(.+?)\s+(?:is(?:,?\s*however,?\s*|\s+\w+)?\s+a\s+(?:major\s+|primary\s+|known\s+|key\s+)?cause\s+of)\s+(.+)",
    r"(.+?)\s+(?:is transcribed from)\s+(.+)",
    r"(.+?)\s+(?:is synthesised from|is synthesized from)\s+(.+)",
]

# Backward causal connectors: [EFFECT] <- [CAUSE]
BACKWARD_CAUSAL_PATTERNS = [
    r"(.+?)\s+(?:is caused by|was caused by)\s+(.+)",
    r"(.+?)\s+(?:results from|resulted from)\s+(.+)",
    r"(.+?)\s+(?:is triggered by|was triggered by)\s+(.+)",
    r"(.+?)\s+(?:is produced by|was produced by)\s+(.+)",
    r"(.+?)\s+(?:is induced by|was induced by)\s+(.+)",
    r"(.+?)\s+(?:originates from|originated from)\s+(.+)",
    r"(.+?)\s+(?:stems from|stemmed from)\s+(.+)",
    r"(.+?)\s+(?:is driven by|was driven by)\s+(.+)",
    r"(.+?)\s+(?:is due to)\s+(.+)",
]


@dataclass
class CausalRelation:
    cause: str
    effect: str
    raw_pattern: str
    is_forward: bool


@dataclass
class CausalDirectionResult:
    claim_relation: Optional[CausalRelation]
    evidence_relation: Optional[CausalRelation]
    is_inversion_detected: bool
    confidence_penalty: float
    explanation: str


class CausalDirectionChecker:
    """Detects directional inversions in cause-and-effect assertions."""

    def extract_causal_relation(self, text: str) -> Optional[CausalRelation]:
        """Extract cause and effect entities from sentence."""
        clean = text.strip()

        # Check backward patterns first (passive voice: 'A is caused by B' -> effect=A, cause=B)
        for pat in BACKWARD_CAUSAL_PATTERNS:
            m = re.search(pat, clean, flags=re.IGNORECASE)
            if m:
                effect = m.group(1).strip()
                cause = m.group(2).strip()
                if len(cause) > 2 and len(effect) > 2:
                    return CausalRelation(
                        cause=cause,
                        effect=effect,
                        raw_pattern=pat,
                        is_forward=False,
                    )

        # Check forward patterns
        for pat in FORWARD_CAUSAL_PATTERNS:
            m = re.search(pat, clean, flags=re.IGNORECASE)
            if m:
                cause = m.group(1).strip()
                effect = m.group(2).strip()
                if len(cause) > 2 and len(effect) > 2:
                    return CausalRelation(
                        cause=cause,
                        effect=effect,
                        raw_pattern=pat,
                        is_forward=True,
                    )

        return None

    def check_inversion(
        self, claim_text: str, evidence_text: str
    ) -> CausalDirectionResult:
        """
        Compare causal direction between claim and evidence sentences.
        """
        c_rel = self.extract_causal_relation(claim_text)
        if not c_rel:
            return CausalDirectionResult(
                claim_relation=None,
                evidence_relation=None,
                is_inversion_detected=False,
                confidence_penalty=0.0,
                explanation="No explicit causal relation detected in claim.",
            )

        ev_sentences = [
            s.strip() for s in re.split(r"[.!?\n]+", evidence_text) if len(s.strip().split()) >= 3
        ]
        if not ev_sentences:
            ev_sentences = [evidence_text]

        # Extract primary subject from first sentence for pronoun resolution
        primary_subject = ""
        if ev_sentences:
            first_sent = ev_sentences[0]
            subj_match = re.match(r"^([A-Za-z0-9\s,\(\)]+?)\s+(?:is|are|was|were)\b", first_sent)
            if subj_match:
                primary_subject = subj_match.group(1).strip()

        e_rel_first = None

        for s in ev_sentences:
            candidate_s = s
            if primary_subject and re.match(r"^(?:It|This|They)\s+(?:is|are|was|were)\b", s, re.IGNORECASE):
                candidate_s = re.sub(r"^(?:It|This|They)\s+", f"{primary_subject} ", s, count=1, flags=re.IGNORECASE)

            e_rel = self.extract_causal_relation(candidate_s)
            if e_rel and e_rel_first is None:
                e_rel_first = e_rel

            if e_rel:
                c_cause_words = set(re.findall(r"\b\w{3,}\b", c_rel.cause.lower()))
                c_effect_words = set(re.findall(r"\b\w{3,}\b", c_rel.effect.lower()))
                e_cause_words = set(re.findall(r"\b\w{3,}\b", e_rel.cause.lower()))
                e_effect_words = set(re.findall(r"\b\w{3,}\b", e_rel.effect.lower()))

                cause_to_effect = len(c_cause_words & e_effect_words) > 0
                effect_to_cause = len(c_effect_words & e_cause_words) > 0

                if cause_to_effect and effect_to_cause:
                    return CausalDirectionResult(
                        claim_relation=c_rel,
                        evidence_relation=e_rel,
                        is_inversion_detected=True,
                        confidence_penalty=0.85,
                        explanation=(
                            f"Causal inversion detected: Claim asserts '{c_rel.cause}' causes '{c_rel.effect}', "
                            f"whereas evidence affirms '{e_rel.cause}' causes '{e_rel.effect}'."
                        ),
                    )

        return CausalDirectionResult(
            claim_relation=c_rel,
            evidence_relation=e_rel_first,
            is_inversion_detected=False,
            confidence_penalty=0.0,
            explanation="Causal entities align in the correct direction." if e_rel_first else "No explicit causal relation detected in evidence to verify against.",
        )
