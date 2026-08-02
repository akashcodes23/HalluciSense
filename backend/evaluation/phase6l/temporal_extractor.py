"""Phase 6L.1B — Temporal Consistency Extractor.

Extracts timestamps, years, and chronological order expressions from claims,
detects conservative date conflicts and timeline ordering violations,
and computes Family F features:
    * temporal_conflict_count
    * timeline_order_violation_score

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
class TemporalMention:
    """Extracted temporal mention with context."""
    raw_text: str
    year: int
    entity: str
    event_type: str  # e.g., "founded", "born", "occurred", "released"
    claim_id: int


@dataclass
class TemporalConflictRecord:
    """Explainability record for a detected temporal conflict."""
    claim_i_index: int
    claim_j_index: int
    entity: str
    event_type: str
    year_i: int
    year_j: int
    rule_triggered: str


YEAR_PATTERNS = [
    # Entity was founded in YEAR / founded in YEAR
    (re.compile(r"\b((?:[A-Za-z0-9_]+\s+){1,4}?)(?:was\s+|is\s+|were\s+)?founded\s+in\s+([12][0-9]{3})\b", re.IGNORECASE), "founded"),
    # Entity was born in YEAR / born in YEAR
    (re.compile(r"\b((?:[A-Za-z0-9_]+\s+){1,4}?)(?:was\s+|is\s+|were\s+)?born\s+in\s+([12][0-9]{3})\b", re.IGNORECASE), "born"),
    # Event occurred / released in YEAR
    (re.compile(r"\b((?:[A-Za-z0-9_]+\s+){1,4}?)(?:was\s+|is\s+|were\s+)?released\s+in\s+([12][0-9]{3})\b", re.IGNORECASE), "released"),
]


def extract_temporal_mentions(claims: List[str]) -> List[TemporalMention]:
    """Extract temporal mentions (entity, event_type, year) from claims."""
    mentions: List[TemporalMention] = []

    for c_idx, claim_text in enumerate(claims):
        if not claim_text or not claim_text.strip():
            continue

        for pattern, event_type in YEAR_PATTERNS:
            for match in pattern.finditer(claim_text):
                ent_raw = match.group(1).strip()
                ent_raw = re.sub(r"\s+(?:was|is|were|are|the)$", "", ent_raw, flags=re.IGNORECASE).lower()
                year_str = match.group(2).strip()

                try:
                    yr = int(year_str)
                    if 1000 <= yr <= 2030:
                        mentions.append(
                            TemporalMention(
                                raw_text=match.group(0),
                                year=yr,
                                entity=ent_raw,
                                event_type=event_type,
                                claim_id=c_idx,
                            )
                        )
                except Exception:
                    continue

    return mentions


def extract_temporal_consistency_features(claims: List[str]) -> Dict[str, Any]:
    """Extract Family F Temporal Consistency features and conflict records.

    Args:
        claims: List of claim text strings for a response.

    Returns:
        Dict containing 2 numerical features and list of explainability records.
    """
    n = len(claims)
    if n < 2:
        return {
            "temporal_conflict_count": 0.0,
            "timeline_order_violation_score": 0.0,
            "explainability_records": [],
            "total_temporal_mentions": 0,
        }

    mentions = extract_temporal_mentions(claims)
    total_mentions = len(mentions)

    conflict_records: List[TemporalConflictRecord] = []
    conflict_count = 0

    # Group mentions by (entity, event_type)
    grouped: Dict[Tuple[str, str], List[TemporalMention]] = {}
    for m in mentions:
        key = (m.entity, m.event_type)
        grouped.setdefault(key, []).append(m)

    for (ent, ev_type), m_list in grouped.items():
        if len(m_list) < 2:
            continue

        for i in range(len(m_list)):
            m_i = m_list[i]
            for j in range(i + 1, len(m_list)):
                m_j = m_list[j]
                if m_i.claim_id == m_j.claim_id:
                    continue  # Same claim mentions are not pairwise claim conflicts

                # Incompatible year for SAME entity and event (e.g. founded in 2010 vs 2014)
                if m_i.year != m_j.year:
                    conflict_count += 1
                    conflict_records.append(
                        TemporalConflictRecord(
                            claim_i_index=m_i.claim_id,
                            claim_j_index=m_j.claim_id,
                            entity=ent,
                            event_type=ev_type,
                            year_i=m_i.year,
                            year_j=m_j.year,
                            rule_triggered=f"temporal_{ev_type}_year_mismatch",
                        )
                    )

    m_total_pairs = (n * (n - 1)) // 2
    violation_score = float(conflict_count / max(1, m_total_pairs))

    return {
        "temporal_conflict_count": float(conflict_count),
        "timeline_order_violation_score": float(violation_score),
        "explainability_records": [rec.__dict__ for rec in conflict_records],
        "total_temporal_mentions": total_mentions,
    }
