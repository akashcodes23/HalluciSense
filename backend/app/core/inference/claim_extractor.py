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
    r"vs\.", r"etc\.", r"St\.", r"Inc\.", r"Ltd\.", r"Co\.", r"Corp\.",
    r"approx\.", r"fig\.", r"eq\.", r"no\.", r"vol\.", r"dept\.", r"et al\."
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

    # Protect abbreviations, decimals, scientific notation, and initials
    protected_ranges: List[tuple[int, int]] = []

    for abbr in ABBREVIATIONS:
        for m in re.finditer(r'\b' + abbr, raw_text, flags=re.IGNORECASE):
            protected_ranges.append((m.start(), m.end()))

    for m in re.finditer(r'\d+\.\d+(?:[eE][+-]?\d+)?', raw_text):
        protected_ranges.append((m.start(), m.end()))

    for m in re.finditer(r'(?:^|\s)[A-Z]\.(?=\s+[A-Z])', raw_text):
        protected_ranges.append((m.end() - 1, m.end()))

    for m in re.finditer(r'\.{3,}', raw_text):
        protected_ranges.append((m.start(), m.end()))

    def is_protected(pos: int) -> bool:
        return any(start <= pos < end for start, end in protected_ranges)

    claims: List[Dict[str, Any]] = []
    claim_id = 0
    current_start = 0

    while current_start < len(raw_text) and raw_text[current_start].isspace():
        current_start += 1

    for m in re.finditer(r'[.!?]+["\'”’\)\]]*(?=\s+|$)', raw_text):
        end_pos = m.end()
        punct_pos = m.start()
        if not is_protected(punct_pos):
            sent = raw_text[current_start:end_pos].strip()
            if sent:
                claims.append({
                    "claim_id": claim_id,
                    "text": sent,
                })
                claim_id += 1
            next_start = end_pos
            while next_start < len(raw_text) and raw_text[next_start].isspace():
                next_start += 1
            current_start = next_start

    if current_start < len(raw_text):
        remaining = raw_text[current_start:].strip()
        if remaining:
            claims.append({
                "claim_id": claim_id,
                "text": remaining,
            })

    if not claims:
        claims = [{"claim_id": 0, "text": raw_text}]

    logger.debug("extract_claims_complete", input_length=len(raw_text), claim_count=len(claims))
    return claims
