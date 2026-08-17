"""Claim Decomposition Engine for HalluciSense Enhanced P1 Pipeline.

Decomposes complex, multi-clause scientific statements into atomic factual propositions
without requiring external LLM calls. Provides multi-strategy aggregation over
proposition-level entailment/contradiction scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence


class AggregationStrategy(str, Enum):
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted-mean"
    MAX_RISK = "max-risk"
    UNSUPPORTED_RATIO = "unsupported-ratio"
    TOP_K_RISK = "top-k-risk"


@dataclass
class AtomicProposition:
    """An atomic factual claim extracted from a larger text."""
    text: str
    index: int
    source_sentence: str
    weight: float = 1.0
    h_score: Optional[float] = None
    is_contradiction: Optional[bool] = None
    evidence_snippet: Optional[str] = None


@dataclass
class DecompositionResult:
    """Result of decomposing and scoring a text statement."""
    original_text: str
    propositions: List[AtomicProposition] = field(default_factory=list)
    aggregated_h_score: float = 0.0
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MAX_RISK
    num_propositions: int = 0
    num_flagged_propositions: int = 0


# Discourse markers and conversational fluff to strip
DISCOURSE_PREFIXES = [
    r"^(it is (well )?known that\s*)",
    r"^(research shows that\s*)",
    r"^(studies have demonstrated that\s*)",
    r"^(scientists believe that\s*)",
    r"^(in fact,\s*)",
    r"^(notably,\s*)",
    r"^(furthermore,\s*)",
    r"^(additionally,\s*)",
    r"^(specifically,\s*)",
    r"^(as a matter of fact,\s*)",
]

# Coordinate conjunctions that often link independent claims
CLAUSE_SPLITTERS = [
    r";\s*",
    r",\s*and\s+(?=[a-z0-9\s]+(?:is|are|was|were|has|have|had|causes|contains|synthesises|predicts|allows|works|means))",
    r",\s*but\s+",
    r",\s*which\s+(?:is|are|was|were|also|means)\s+",
    r",\s*while\s+",
    r",\s*whereas\s+",
]


class ClaimDecomposer:
    """Splits complex sentences into verifiable atomic propositions."""

    def __init__(self, min_length: int = 15, max_propositions_per_sentence: int = 5):
        self.min_length = min_length
        self.max_propositions_per_sentence = max_propositions_per_sentence

    def clean_text(self, text: str) -> str:
        """Strip boilerplate conversational framing."""
        cleaned = text.strip()
        for pat in DISCOURSE_PREFIXES:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation boundaries."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw_sentences if len(s.strip()) >= 5]

    def decompose_sentence(self, sentence: str) -> List[str]:
        """Decompose a single sentence into atomic propositions."""
        cleaned = self.clean_text(sentence)
        if not cleaned:
            return []

        # Try splitting on coordinate clause delimiters
        fragments = [cleaned]
        for splitter in CLAUSE_SPLITTERS:
            new_fragments = []
            for frag in fragments:
                parts = re.split(splitter, frag, flags=re.IGNORECASE)
                for p in parts:
                    p = p.strip()
                    if p:
                        new_fragments.append(p)
            fragments = new_fragments

        propositions: List[str] = []
        for frag in fragments:
            frag = frag.rstrip(".,;")
            if len(frag) >= self.min_length:
                # Ensure it starts capitalized
                frag_norm = frag[0].upper() + frag[1:] if len(frag) > 1 else frag.upper()
                propositions.append(frag_norm)

        # Fallback to the original cleaned sentence if splitting broke it too small
        if not propositions and len(cleaned) >= 5:
            propositions = [cleaned]

        return propositions[: self.max_propositions_per_sentence]

    def decompose(self, text: str) -> List[AtomicProposition]:
        """Decompose full text into a list of AtomicProposition dataclasses."""
        sentences = self.split_sentences(text)
        all_props: List[AtomicProposition] = []
        idx = 0

        for sent in sentences:
            sub_claims = self.decompose_sentence(sent)
            for sub in sub_claims:
                idx += 1
                # Weigh shorter subordinate elaborations slightly higher for risk detection
                weight = 1.0
                all_props.append(
                    AtomicProposition(
                        text=sub,
                        index=idx,
                        source_sentence=sent,
                        weight=weight,
                    )
                )

        return all_props

    @staticmethod
    def aggregate_scores(
        propositions: Sequence[AtomicProposition],
        strategy: AggregationStrategy = AggregationStrategy.MAX_RISK,
        top_k: int = 2,
        contradiction_threshold: float = 0.50,
    ) -> float:
        """Aggregate proposition-level scores into an overall claim hallucination score."""
        scores = [p.h_score for p in propositions if p.h_score is not None]
        if not scores:
            return 0.5

        if strategy == AggregationStrategy.MEAN:
            return float(sum(scores) / len(scores))

        elif strategy == AggregationStrategy.WEIGHTED_MEAN:
            weights = [p.weight for p in propositions if p.h_score is not None]
            total_w = sum(weights)
            if total_w == 0:
                return float(sum(scores) / len(scores))
            return float(sum(s * w for s, w in zip(scores, weights)) / total_w)

        elif strategy == AggregationStrategy.MAX_RISK:
            return float(max(scores))

        elif strategy == AggregationStrategy.UNSUPPORTED_RATIO:
            flagged = sum(1 for s in scores if s >= contradiction_threshold)
            return float(flagged / len(scores))

        elif strategy == AggregationStrategy.TOP_K_RISK:
            sorted_scores = sorted(scores, reverse=True)
            k_scores = sorted_scores[: min(top_k, len(sorted_scores))]
            return float(sum(k_scores) / len(k_scores))

        return float(max(scores))
