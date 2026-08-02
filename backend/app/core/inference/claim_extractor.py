"""Production Claim Extractor Module.

Performs robust sentence segmentation, claim ID assignment, and ordering preservation.
Does NOT rely on simple split('.') logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)

# Common English abbreviations to protect against premature sentence splitting
ABBREVIATIONS = [
    r"Dr\.", r"Mr\.", r"Mrs\.", r"Ms\.", r"Prof\.", r"Sr\.", r"Jr\.",
    r"U\.S\.", r"U\.K\.", r"E\.U\.", r"i\.e\.", r"e\.g\.", r"a\.m\.", r"p\.m\.",
    r"vs\.", r"etc\.", r"St\.", r"Inc\.", r"Ltd\.", r"Co\.", r"Corp\."
]

def extract_claims(text: str) -> List[Dict[str, Any]]:
    """Extract atomic claims from raw text response.

    Args:
        text: Raw response string.

    Returns:
        List of dicts containing claim_id (int) and text (str).
    """
    if not text or not text.strip():
        return []

    raw_text = text.strip()

    # Protect abbreviations by temporarily replacing period with placeholder token
    protected_text = raw_text
    abbrev_map = {}
    for idx, abbr in enumerate(ABBREVIATIONS):
        token = f"__ABBR_{idx}__"
        matches = re.findall(abbr, protected_text, flags=re.IGNORECASE)
        for m in set(matches):
            abbrev_map[token] = m
            protected_text = protected_text.replace(m, token)

    # Sentence boundary regex: split on [.!?] followed by whitespace or newline
    sentence_pattern = r"(?<=[.!?])\s+"
    raw_sentences = re.split(sentence_pattern, protected_text)

    claims = []
    claim_id = 0

    for sent in raw_sentences:
        # Restore protected abbreviations
        clean_sent = sent
        for token, original in abbrev_map.items():
            clean_sent = clean_sent.replace(token, original)

        clean_sent = clean_sent.strip()
        if len(clean_sent) > 0:
            claims.append({
                "claim_id": claim_id,
                "text": clean_sent,
            })
            claim_id += 1

    # Fallback to full text if segmentation yields zero claims
    if not claims:
        claims = [{"claim_id": 0, "text": raw_text}]

    logger.debug("extract_claims_complete", input_length=len(raw_text), claim_count=len(claims))
    return claims
