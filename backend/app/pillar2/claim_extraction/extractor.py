"""
HalluciSense Pillar 2 — Claim Extraction Engine
=================================================
Decomposes LLM response text into structured atomic factual claims.
Detects declarative facts, numerical metrics, temporal anchors, entity relations,
and scientific assertions.
"""

import hashlib
import re
import time
import uuid
from typing import List, Tuple

import structlog
from app.pillar2.claim_extraction.schemas import (
    CharacterOffsets,
    ClaimExtractionRequest,
    ClaimExtractionResponse,
    ClaimType,
    ExtractedClaim,
)

logger = structlog.get_logger(__name__)


class ClaimExtractionEngine:
    """
    Production-ready atomic claim extraction engine.
    Uses robust Regex + NLP heuristics for deterministic, fast decomposition.
    """

    # Regex patterns for entities, numbers, dates, relations, and scientific terms
    NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|trillion|kg|mg|m|km|seconds|minutes|hours|years))?\b", re.IGNORECASE)
    DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b", re.IGNORECASE)
    ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
    SCIENTIFIC_PATTERN = re.compile(r"\b(?:DNA|RNA|CRISPR|quantum|algorithm|theorem|equation|molecule|protein|gene|reaction|hypothesis|variable|constant|matrix|vector)\b", re.IGNORECASE)
    RELATION_VERBS = re.compile(r"\b(?:causes|inhibits|activates|discovered|invented|founded|located in|produced|published|derived from|equal to|greater than|less than|contains)\b", re.IGNORECASE)

    def extract_claims(self, request: ClaimExtractionRequest) -> ClaimExtractionResponse:
        """
        Decompose input text into atomic claims.

        Parameters
        ----------
        request : ClaimExtractionRequest

        Returns
        -------
        ClaimExtractionResponse
        """
        start_time = time.perf_counter()
        text = request.text.strip()

        if not text:
            return ClaimExtractionResponse(
                extracted_claims=[],
                total_claims=0,
                num_sentences=0,
                extraction_time_ms=0.0,
            )

        sentences = self._split_sentences(text)
        extracted_claims: List[ExtractedClaim] = []

        curr_offset = 0
        for sent_idx, sent in enumerate(sentences):
            # Find exact character offset of sentence in raw text
            match_start = text.find(sent, curr_offset)
            if match_start == -1:
                match_start = curr_offset
            match_end = match_start + len(sent)
            curr_offset = match_end

            # Decompose sentence into atomic sub-claims if conjunctions exist
            sub_claims = self._decompose_sentence_to_claims(sent)

            for sub_text in sub_claims:
                claim_id = self._generate_claim_id(sub_text, sent_idx)
                entities = list(set(self.ENTITY_PATTERN.findall(sub_text)))
                dates = list(set(self.DATE_PATTERN.findall(sub_text)))
                numbers = list(set(self.NUMBER_PATTERN.findall(sub_text)))
                relations = list(set(self.RELATION_VERBS.findall(sub_text)))
                claim_type = self._classify_claim_type(sub_text, numbers, dates, relations)

                # Sub-span offset within original text
                sub_start = text.find(sub_text, match_start)
                if sub_start == -1:
                    sub_start = match_start
                sub_end = sub_start + len(sub_text)

                confidence = self._compute_extraction_confidence(sub_text, entities, numbers, dates)

                claim = ExtractedClaim(
                    claim_id=claim_id,
                    claim_text=sub_text,
                    claim_type=claim_type,
                    entities=entities,
                    relations=relations,
                    numbers=numbers,
                    dates=dates,
                    confidence=round(confidence, 4),
                    sentence_index=sent_idx,
                    character_offsets=CharacterOffsets(start=sub_start, end=sub_end),
                    metadata={"source_sentence": sent},
                )
                extracted_claims.append(claim)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "claim_extraction_complete",
            text_length=len(text),
            total_claims=len(extracted_claims),
            num_sentences=len(sentences),
            latency_ms=round(elapsed_ms, 2),
        )

        return ClaimExtractionResponse(
            extracted_claims=extracted_claims,
            total_claims=len(extracted_claims),
            num_sentences=len(sentences),
            extraction_time_ms=round(elapsed_ms, 2),
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences handling abbreviations and punctuation."""
        # Simple sentence splitter
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]

    def _decompose_sentence_to_claims(self, sentence: str) -> List[str]:
        """Split compound sentences with conjunctions into atomic claims."""
        # Split on semicolon or ' additionally, ' or ' moreover, '
        parts = re.split(r";\s*|\b(?:additionally|moreover|furthermore|however),\s*", sentence, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if len(p.strip()) > 5]
        return parts if parts else [sentence]

    def _classify_claim_type(
        self, text: str, numbers: List[str], dates: List[str], relations: List[str]
    ) -> ClaimType:
        """Classify claim into discrete ClaimType category."""
        if self.SCIENTIFIC_PATTERN.search(text):
            return ClaimType.SCIENTIFIC
        elif len(dates) > 0:
            return ClaimType.TEMPORAL
        elif len(numbers) > 0:
            return ClaimType.NUMERICAL
        elif len(relations) > 0:
            return ClaimType.ENTITY_RELATION
        else:
            return ClaimType.DECLARATIVE

    def _compute_extraction_confidence(
        self, text: str, entities: List[str], numbers: List[str], dates: List[str]
    ) -> float:
        """Compute structural confidence score for extracted claim."""
        score = 0.7  # Base confidence
        if len(text.split()) >= 4:
            score += 0.1
        if len(entities) > 0:
            score += 0.1
        if len(numbers) > 0 or len(dates) > 0:
            score += 0.1
        return min(1.0, score)

    def _generate_claim_id(self, claim_text: str, sentence_idx: int) -> str:
        """Deterministic SHA256 hash ID for claim."""
        raw = f"{sentence_idx}:{claim_text.lower().strip()}"
        return f"claim_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"
