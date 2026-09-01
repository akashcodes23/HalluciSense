"""Phase 42 — Deterministic Claim Type Classifier.

Classifies atomic claims into structured verification modalities:
- ARITHMETIC (e.g. "12 x 8 = 96", "100 / 4 is 25")
- UNIT_CONVERSION (e.g. "100 km/h is 27.78 m/s", "1 hour is 60 minutes")
- TEMPORAL_MATH (e.g. "2024 is 4 years after 2020", "January 2020 was before February 2021")
- TEXTUAL_FACT (Empirical encyclopedic statements requiring Wikipedia retrieval)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple


ARITHMETIC_PATTERN = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*(?:[\+\-\*\/x×÷]|plus|minus|times|multiplied by|divided by)\s*(\d+(?:\.\d+)?)\s*(?:=|equals|is equal to|is)\s*(\d+(?:\.\d+)?)\b"
)

PERCENTAGE_PATTERN = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(\d+(?:\.\d+)?)\s*(?:=|equals|is)\s*(\d+(?:\.\d+)?)\b"
)

UNIT_PATTERN = re.compile(
    r"(?i)\b(\d+(?:\.\d+)?)\s*(km/h|m/s|mph|km|m|cm|mm|miles?|meters?|kilometers?|kg|g|lbs?|pounds?|hours?|mins?|minutes?|secs?|seconds?|celsius|fahrenheit|kelvin|°C|°F|K)\s*(?:=|equals|is|is equal to|converted to)\s*(\d+(?:\.\d+)?)\s*(km/h|m/s|mph|km|m|cm|mm|miles?|meters?|kilometers?|kg|g|lbs?|pounds?|hours?|mins?|minutes?|secs?|seconds?|celsius|fahrenheit|kelvin|°C|°F|K)\b"
)

TEMPORAL_PATTERN = re.compile(
    r"(?i)\b(\d{4})\s*(?:was|is)?\s*(\d+)\s*(?:years?|months?|decades?)\s*(?:after|before|later than|earlier than)\s*(\d{4})\b"
)


class ClaimTypeClassifier:
    """Deterministic, regex-driven claim type classifier."""

    @staticmethod
    def classify(claim_text: str) -> Dict[str, Any]:
        text = claim_text.strip()
        
        # 1. Arithmetic
        m_arith = ARITHMETIC_PATTERN.search(text)
        if m_arith:
            return {
                "claim_type": "ARITHMETIC",
                "modality": "symbolic_computation",
                "extracted_entities": {
                    "left": m_arith.group(1),
                    "right": m_arith.group(2),
                    "claimed": m_arith.group(3),
                    "raw_match": m_arith.group(0),
                },
                "confidence": 1.0,
            }
            
        m_pct = PERCENTAGE_PATTERN.search(text)
        if m_pct:
            return {
                "claim_type": "ARITHMETIC",
                "modality": "symbolic_computation",
                "extracted_entities": {
                    "pct": m_pct.group(1),
                    "base": m_pct.group(2),
                    "claimed": m_pct.group(3),
                    "raw_match": m_pct.group(0),
                },
                "confidence": 1.0,
            }

        # 2. Unit Conversion
        m_unit = UNIT_PATTERN.search(text)
        if m_unit:
            return {
                "claim_type": "UNIT_CONVERSION",
                "modality": "symbolic_computation",
                "extracted_entities": {
                    "val_from": m_unit.group(1),
                    "unit_from": m_unit.group(2).lower(),
                    "val_to": m_unit.group(3),
                    "unit_to": m_unit.group(4).lower(),
                    "raw_match": m_unit.group(0),
                },
                "confidence": 1.0,
            }

        # 3. Temporal Math
        m_temp = TEMPORAL_PATTERN.search(text)
        if m_temp:
            return {
                "claim_type": "TEMPORAL_MATH",
                "modality": "symbolic_computation",
                "extracted_entities": {
                    "year_target": m_temp.group(1),
                    "delta": m_temp.group(2),
                    "year_base": m_temp.group(3),
                    "raw_match": m_temp.group(0),
                },
                "confidence": 1.0,
            }

        # 4. Default: Textual Fact
        return {
            "claim_type": "TEXTUAL_FACT",
            "modality": "retrieval_and_nli",
            "extracted_entities": {},
            "confidence": 1.0,
        }
