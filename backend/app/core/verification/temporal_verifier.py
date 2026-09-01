"""Phase 42 — Temporal Math & Date Verifier.

Deterministically verifies calendar differences, years before/after, and basic date arithmetic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def evaluate_temporal_claim(claim_text: str) -> Optional[Dict[str, Any]]:
    """Evaluates relative temporal calculations (e.g. '2024 is 4 years after 2020')."""
    pattern = re.compile(
        r"(?i)\b(\d{4})\s*(?:was|is)?\s*(\d+)\s*(years?|months?|decades?)\s*(after|before|later than|earlier than)\s*(\d{4})\b"
    )
    m = pattern.search(claim_text.strip())
    if not m:
        return None
        
    y_target = int(m.group(1))
    delta = int(m.group(2))
    unit = m.group(3).lower()
    direction = m.group(4).lower()
    y_base = int(m.group(5))
    
    if "decade" in unit:
        delta_years = delta * 10
    elif "month" in unit:
        delta_years = delta / 12.0
    else:
        delta_years = float(delta)
        
    if direction in ("after", "later than"):
        expected_target = y_base + delta_years
    else:
        expected_target = y_base - delta_years
        
    consistent = abs(expected_target - y_target) < 1e-4
    return {
        "verified": True,
        "operation": "temporal_delta",
        "y_target": y_target,
        "y_base": y_base,
        "delta": delta,
        "direction": direction,
        "expected_target": round(expected_target, 1),
        "is_consistent": consistent,
        "explanation": f"{delta} {unit} {direction} {y_base} is {expected_target:.0f} (Claim stated {y_target})",
    }
