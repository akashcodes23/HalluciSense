"""Phase 6L.1B — Numerical Consistency Extractor.

Extracts context-matched numerical quantities (value, magnitude, context, unit)
from claims, detects incompatible numeric values for identical semantic contexts,
and computes Family E features:
    * numeric_conflict_count
    * numeric_conflict_ratio
    * max_numeric_disagreement

Strict Data Firewall Rule:
    * Label-free: No rule or threshold depends on ground truth target y.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class NumericMention:
    """Extracted numerical mention with context."""
    raw_value_str: str
    numeric_value: float
    context: str         # e.g., "capacity_kwh", "revenue_usd", "population", "percentage"
    claim_id: int
    raw_text: str


@dataclass
class NumericConflictRecord:
    """Explainability record for a detected numerical conflict."""
    claim_i_index: int
    claim_j_index: int
    normalized_context: str
    raw_value_i: str
    raw_value_j: str
    normalized_value_i: float
    normalized_value_j: float
    relative_difference: float
    rule_triggered: str


# Magnitude suffix map
MAGNITUDE_MAP = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "b": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

NUMERIC_CONTEXT_PATTERNS = [
    # Battery capacity: 75 kWh / 75kwh
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:kwh|kilowatt[- ]hours?)", re.IGNORECASE), "battery_capacity_kwh", 1.0),
    # Currency: $10M / $10 million / 10 million dollars
    (re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(k|m|b|t|thousand|million|billion|trillion)?", re.IGNORECASE), "currency_usd", None),
    # Percentage: 50% / 50 percent
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE), "percentage", 1.0),
    # Power: 150 kW / 150kw
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:kw|kilowatts?)\b(?!h)", re.IGNORECASE), "power_kw", 1.0),
    # Population / count: 500 employees / 500 people
    (re.compile(r"(\d+(?:\.\d+)?)\s*(k|m|b|thousand|million|billion)?\s+(?:employees|people|users|subscribers|members)", re.IGNORECASE), "population_count", None),
]


def parse_numeric_value(num_str: str, mult_str: Optional[str] = None) -> float:
    """Parse numeric string and scale by magnitude suffix if present."""
    val = float(num_str)
    if mult_str:
        mult_key = mult_str.lower().strip()
        val *= MAGNITUDE_MAP.get(mult_key, 1.0)
    return val


def extract_numeric_mentions(claims: List[str]) -> List[NumericMention]:
    """Extract numeric mentions and context from claims."""
    mentions: List[NumericMention] = []

    for c_idx, claim_text in enumerate(claims):
        if not claim_text or not claim_text.strip():
            continue

        for pattern, context_name, default_mult in NUMERIC_CONTEXT_PATTERNS:
            for match in pattern.finditer(claim_text):
                raw_num = match.group(1)
                mult_grp = match.group(2) if match.lastindex >= 2 else None

                try:
                    num_val = parse_numeric_value(raw_num, mult_grp)
                    mentions.append(
                        NumericMention(
                            raw_value_str=match.group(0),
                            numeric_value=num_val,
                            context=context_name,
                            claim_id=c_idx,
                            raw_text=claim_text,
                        )
                    )
                except Exception:
                    continue

    return mentions


def extract_numeric_consistency_features(claims: List[str]) -> Dict[str, Any]:
    """Extract Family E Numerical Consistency features and conflict records.

    Args:
        claims: List of claim text strings for a response.

    Returns:
        Dict containing 3 numerical features and list of explainability records.
    """
    n = len(claims)
    if n < 2:
        return {
            "numeric_conflict_count": 0.0,
            "numeric_conflict_ratio": 0.0,
            "max_numeric_disagreement": 0.0,
            "explainability_records": [],
            "total_numeric_mentions": 0,
        }

    mentions = extract_numeric_mentions(claims)
    total_mentions = len(mentions)

    conflict_records: List[NumericConflictRecord] = []
    conflict_count = 0
    max_rel_diff = 0.0

    # Group mentions by context
    grouped: Dict[str, List[NumericMention]] = {}
    for m in mentions:
        grouped.setdefault(m.context, []).append(m)

    for context_name, m_list in grouped.items():
        if len(m_list) < 2:
            continue

        for i in range(len(m_list)):
            m_i = m_list[i]
            for j in range(i + 1, len(m_list)):
                m_j = m_list[j]
                if m_i.claim_id == m_j.claim_id:
                    continue  # Same claim mentions are not pairwise claim conflicts

                v_i, v_j = m_i.numeric_value, m_j.numeric_value
                denom = max(abs(v_i), abs(v_j))

                if denom > 0:
                    rel_diff = abs(v_i - v_j) / denom
                else:
                    rel_diff = 0.0

                # Numerical collision: relative difference > 0.01 (1%) for SAME semantic context
                if rel_diff > 0.01:
                    conflict_count += 1
                    if rel_diff > max_rel_diff:
                        max_rel_diff = rel_diff

                    conflict_records.append(
                        NumericConflictRecord(
                            claim_i_index=m_i.claim_id,
                            claim_j_index=m_j.claim_id,
                            normalized_context=context_name,
                            raw_value_i=m_i.raw_value_str,
                            raw_value_j=m_j.raw_value_str,
                            normalized_value_i=v_i,
                            normalized_value_j=v_j,
                            relative_difference=float(rel_diff),
                            rule_triggered=f"numeric_{context_name}_disagreement",
                        )
                    )

    m_total_pairs = (n * (n - 1)) // 2
    conflict_ratio = float(conflict_count / max(1, m_total_pairs))

    return {
        "numeric_conflict_count": float(conflict_count),
        "numeric_conflict_ratio": float(conflict_ratio),
        "max_numeric_disagreement": float(max_rel_diff),
        "explainability_records": [rec.__dict__ for rec in conflict_records],
        "total_numeric_mentions": total_mentions,
    }
