"""Phase 42 — Unit & Dimension Conversion Verifier.

Deterministically verifies physical and temporal unit conversions with floating point tolerances.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


# Canonical conversion factors to SI base units
LENGTH_TO_METERS = {
    "m": 1.0, "meters": 1.0, "meter": 1.0,
    "km": 1000.0, "kilometers": 1000.0, "kilometer": 1000.0,
    "cm": 0.01, "centimeters": 0.01, "centimeter": 0.01,
    "mm": 0.001, "millimeters": 0.001, "millimeter": 0.001,
    "miles": 1609.344, "mile": 1609.344,
}

TIME_TO_SECONDS = {
    "s": 1.0, "seconds": 1.0, "second": 1.0, "secs": 1.0, "sec": 1.0,
    "min": 60.0, "mins": 60.0, "minutes": 60.0, "minute": 60.0,
    "h": 3600.0, "hours": 3600.0, "hour": 3600.0,
    "days": 86400.0, "day": 86400.0,
}

SPEED_TO_MS = {
    "m/s": 1.0,
    "km/h": 1.0 / 3.6,
    "mph": 0.44704,
}

MASS_TO_KG = {
    "kg": 1.0, "kilograms": 1.0, "kilogram": 1.0,
    "g": 0.001, "grams": 0.001, "gram": 0.001,
    "lbs": 0.453592, "pounds": 0.453592, "pound": 0.453592,
}


def evaluate_unit_claim(claim_text: str) -> Optional[Dict[str, Any]]:
    """Evaluates physical unit conversion statements."""
    pattern = re.compile(
        r"(?i)(\d+(?:\.\d+)?)\s*([a-zA-Z/°]+)\s*(?:=|equals|is|is equal to)\s*(\d+(?:\.\d+)?)\s*([a-zA-Z/°]+)"
    )
    m = pattern.search(claim_text.strip())
    if not m:
        return None
        
    val_from = float(m.group(1))
    unit_from = m.group(2).lower()
    val_to = float(m.group(3))
    unit_to = m.group(4).lower()
    
    # 1. Speed
    if unit_from in SPEED_TO_MS and unit_to in SPEED_TO_MS:
        base_si = val_from * SPEED_TO_MS[unit_from]
        expected_to = base_si / SPEED_TO_MS[unit_to]
        consistent = abs(expected_to - val_to) / max(expected_to, 1e-4) < 0.02
        return {
            "verified": True,
            "dimension": "speed",
            "val_from": val_from,
            "unit_from": unit_from,
            "expected_val": round(expected_to, 4),
            "claimed_val": round(val_to, 4),
            "unit_to": unit_to,
            "is_consistent": consistent,
            "explanation": f"{val_from} {unit_from} = {expected_to:.2f} {unit_to} (Claim stated {val_to} {unit_to})",
        }

    # 2. Length
    if unit_from in LENGTH_TO_METERS and unit_to in LENGTH_TO_METERS:
        base_si = val_from * LENGTH_TO_METERS[unit_from]
        expected_to = base_si / LENGTH_TO_METERS[unit_to]
        consistent = abs(expected_to - val_to) / max(expected_to, 1e-4) < 0.01
        return {
            "verified": True,
            "dimension": "length",
            "val_from": val_from,
            "unit_from": unit_from,
            "expected_val": round(expected_to, 4),
            "claimed_val": round(val_to, 4),
            "unit_to": unit_to,
            "is_consistent": consistent,
            "explanation": f"{val_from} {unit_from} = {expected_to:.2f} {unit_to} (Claim stated {val_to} {unit_to})",
        }

    # 3. Time
    if unit_from in TIME_TO_SECONDS and unit_to in TIME_TO_SECONDS:
        base_si = val_from * TIME_TO_SECONDS[unit_from]
        expected_to = base_si / TIME_TO_SECONDS[unit_to]
        consistent = abs(expected_to - val_to) / max(expected_to, 1e-4) < 0.01
        return {
            "verified": True,
            "dimension": "time",
            "val_from": val_from,
            "unit_from": unit_from,
            "expected_val": round(expected_to, 4),
            "claimed_val": round(val_to, 4),
            "unit_to": unit_to,
            "is_consistent": consistent,
            "explanation": f"{val_from} {unit_from} = {expected_to:.2f} {unit_to} (Claim stated {val_to} {unit_to})",
        }

    # 4. Mass
    if unit_from in MASS_TO_KG and unit_to in MASS_TO_KG:
        base_si = val_from * MASS_TO_KG[unit_from]
        expected_to = base_si / MASS_TO_KG[unit_to]
        consistent = abs(expected_to - val_to) / max(expected_to, 1e-4) < 0.01
        return {
            "verified": True,
            "dimension": "mass",
            "val_from": val_from,
            "unit_from": unit_from,
            "expected_val": round(expected_to, 4),
            "claimed_val": round(val_to, 4),
            "unit_to": unit_to,
            "is_consistent": consistent,
            "explanation": f"{val_from} {unit_from} = {expected_to:.2f} {unit_to} (Claim stated {val_to} {unit_to})",
        }

    return None
