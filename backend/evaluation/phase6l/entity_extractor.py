"""Phase 6L.1B — Entity Consistency Extractor.

Extracts named entities and asserted attributes/predicates from claim texts,
detects conservative intra-response entity attribute conflicts, and computes Family D features:
    * entity_conflict_count
    * entity_conflict_ratio
    * entity_attribute_disagreement_score

Strict Data Firewall Rule:
    * Label-free: No rule or threshold depends on ground truth target y.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EntityMention:
    """Extracted entity mention with attribute context."""
    surface_form: str
    normalized_entity: str
    attribute_type: str  # e.g., "born_in", "founded_in", "located_in", "capital_of", "status"
    value: str
    claim_id: int
    raw_text: str


@dataclass
class EntityConflictRecord:
    """Explainability record for a detected entity conflict."""
    claim_i_index: int
    claim_j_index: int
    entity: str
    attribute_type: str
    value_i: str
    value_j: str
    rule_triggered: str


# Pattern rules for conservative entity attribute extraction
ENTITY_ATTRIBUTE_PATTERNS = [
    # Birthplace: X was born in Y / X born in Y
    (re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was\s+)?born\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE), "born_in"),
    # Founding: X was founded in Y / X founded in Y
    (re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was\s+)?founded\s+in\s+([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)", re.IGNORECASE), "founded_in"),
    # Location: X is located in Y / X located in Y
    (re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?located\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE), "located_in"),
    # Capital: Y is the capital of X / capital of X is Y
    (re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is\s+)?the\s+capital\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE), "capital_of"),
    # Founder: X was founded by Y / X founded by Y
    (re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was\s+)?founded\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE), "founded_by"),
]


def normalize_entity_string(text: str) -> str:
    """Normalize entity text: lowercase, strip punctuation and whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_entity_mentions(claims: List[str]) -> List[EntityMention]:
    """Extract entity mentions and attribute values from a list of claims."""
    mentions: List[EntityMention] = []

    for c_idx, claim_text in enumerate(claims):
        if not claim_text or not claim_text.strip():
            continue

        for pattern, attr_type in ENTITY_ATTRIBUTE_PATTERNS:
            for match in pattern.finditer(claim_text):
                ent_raw = match.group(1).strip()
                val_raw = match.group(2).strip()

                # Clean trailing auxiliary verbs
                ent_raw = re.sub(r"\s+(?:was|is|were|are|the)$", "", ent_raw, flags=re.IGNORECASE)
                val_raw = re.sub(r"\s+(?:was|is|were|are|the)$", "", val_raw, flags=re.IGNORECASE)

                norm_ent = normalize_entity_string(ent_raw)
                norm_val = normalize_entity_string(val_raw)

                if norm_ent and norm_val:
                    mentions.append(
                        EntityMention(
                            surface_form=ent_raw,
                            normalized_entity=norm_ent,
                            attribute_type=attr_type,
                            value=norm_val,
                            claim_id=c_idx,
                            raw_text=claim_text,
                        )
                    )

    return mentions


def extract_entity_consistency_features(claims: List[str]) -> Dict[str, Any]:
    """Extract Family D Entity Consistency features and conflict records.

    Args:
        claims: List of claim text strings for a response.

    Returns:
        Dict containing 3 numerical features and list of explainability records.
    """
    n = len(claims)
    if n < 2:
        return {
            "entity_conflict_count": 0.0,
            "entity_conflict_ratio": 0.0,
            "entity_attribute_disagreement_score": 0.0,
            "explainability_records": [],
            "total_entities_detected": 0,
        }

    mentions = extract_entity_mentions(claims)
    unique_entities = set(m.normalized_entity for m in mentions)
    num_entities = len(unique_entities)

    conflict_records: List[EntityConflictRecord] = []
    conflict_count = 0

    # Group mentions by (normalized_entity, attribute_type)
    grouped: Dict[Tuple[str, str], List[EntityMention]] = {}
    for m in mentions:
        key = (m.normalized_entity, m.attribute_type)
        grouped.setdefault(key, []).append(m)

    for (norm_ent, attr_type), m_list in grouped.items():
        if len(m_list) < 2:
            continue

        for i in range(len(m_list)):
            m_i = m_list[i]
            for j in range(i + 1, len(m_list)):
                m_j = m_list[j]
                if m_i.claim_id == m_j.claim_id:
                    continue  # Intra-claim repeats are not pairwise claim conflicts

                # Incompatible attributes for SAME entity and attribute type
                if m_i.value != m_j.value:
                    conflict_count += 1
                    conflict_records.append(
                        EntityConflictRecord(
                            claim_i_index=m_i.claim_id,
                            claim_j_index=m_j.claim_id,
                            entity=m_i.normalized_entity,
                            attribute_type=attr_type,
                            value_i=m_i.value,
                            value_j=m_j.value,
                            rule_triggered=f"incompatible_{attr_type}_values",
                        )
                    )

    m_total_pairs = (n * (n - 1)) // 2
    conflict_ratio = float(conflict_count / max(1, num_entities)) if num_entities > 0 else 0.0
    disagreement_score = float(conflict_count / max(1, m_total_pairs))

    return {
        "entity_conflict_count": float(conflict_count),
        "entity_conflict_ratio": float(conflict_ratio),
        "entity_attribute_disagreement_score": float(disagreement_score),
        "explainability_records": [rec.__dict__ for rec in conflict_records],
        "total_entities_detected": num_entities,
    }
