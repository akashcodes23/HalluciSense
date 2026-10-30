"""Phase 42 — Unit & Dimension Conversion Verifier.

Deterministically verifies physical, temporal, and thermodynamic unit conversions with floating point tolerances.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


# Canonical conversion factors to SI base units
LENGTH_TO_METERS = {
    "m": 1.0, "meters": 1.0, "meter": 1.0, "metres": 1.0, "metre": 1.0,
    "km": 1000.0, "kilometers": 1000.0, "kilometer": 1000.0, "kilometres": 1000.0, "kilometre": 1000.0,
    "cm": 0.01, "centimeters": 0.01, "centimeter": 0.01, "centimetres": 0.01, "centimetre": 0.01,
    "mm": 0.001, "millimeters": 0.001, "millimeter": 0.001, "millimetres": 0.001, "millimetre": 0.001,
    "miles": 1609.344, "mile": 1609.344,
}

TIME_TO_SECONDS = {
    "s": 1.0, "seconds": 1.0, "second": 1.0, "secs": 1.0, "sec": 1.0,
    "min": 60.0, "mins": 60.0, "minutes": 60.0, "minute": 60.0,
    "h": 3600.0, "hours": 3600.0, "hour": 3600.0, "hrs": 3600.0, "hr": 3600.0,
    "days": 86400.0, "day": 86400.0,
    "years": 31557600.0, "year": 31557600.0, "yr": 31557600.0,
}

SPEED_TO_MS = {
    "m/s": 1.0, "m⋅s−1": 1.0, "m·s-1": 1.0, "ms-1": 1.0,
    "meters per second": 1.0, "meter per second": 1.0,
    "metres per second": 1.0, "metre per second": 1.0,
    "km/s": 1000.0, "kms-1": 1000.0,
    "km/h": 1.0 / 3.6, "kmh-1": 1.0 / 3.6,
    "mph": 0.44704,
}

MASS_TO_KG = {
    "kg": 1.0, "kilograms": 1.0, "kilogram": 1.0,
    "g": 0.001, "grams": 0.001, "gram": 0.001,
    "mg": 1e-6, "milligrams": 1e-6, "milligram": 1e-6,
    "lbs": 0.453592, "pounds": 0.453592, "pound": 0.453592,
}

MOLAR_MASS_TO_KG_PER_MOL = {
    "g/mol": 0.001, "g·mol-1": 0.001, "g⋅mol−1": 0.001, "gmol-1": 0.001,
    "grams per mole": 0.001, "gram per mole": 0.001,
    "kg/mol": 1.0, "kg·mol-1": 1.0, "kg⋅mol−1": 1.0, "kgmol-1": 1.0,
}

RECOGNIZED_UNITS_REGEX = (
    r"(?:degrees?\s+Celsius|degrees?\s+Fahrenheit|°C|°F|deg\s*C|deg\s*F|Kelvin|K|"
    r"m[⋅·]s[−-]1|meters?\s+per\s+second|metres?\s+per\s+second|m/s|km/s|km/h|mph|"
    r"g/mol|g[⋅·]mol[−-]1|gmol-1|kg/mol|grams?\s+per\s+mole|gram\s+per\s+mole|"
    r"km|m|cm|mm|miles?|meters?|metres?|kilometers?|kilometres?|"
    r"kg|g|mg|pounds?|lbs?|grams?|kilograms?|"
    r"seconds?|secs?|s|minutes?|mins?|min|hours?|hrs?|h|days?|years?|"
    r"Hz|kHz|MHz|GHz|"
    r"J|kJ|MJ|calories?|kcal|watts?|W|kW|MW)"
)


def to_si_quantity(val: float, unit_str: str) -> Tuple[Optional[str], Optional[float]]:
    """Normalise quantity and recognized unit to canonical SI base dimension and value."""
    u = unit_str.lower().strip()
    u = u.replace("⋅", "").replace("·", "").replace("−", "-").replace("^", "")

    # 1. Temperature (SI Base: Kelvin)
    if u in ("celsius", "degrees celsius", "degree celsius", "°c", "deg c"):
        return "temperature", val + 273.15
    if u in ("fahrenheit", "degrees fahrenheit", "degree fahrenheit", "°f", "deg f"):
        return "temperature", (val - 32.0) * (5.0 / 9.0) + 273.15
    if u in ("kelvin", "k"):
        return "temperature", val

    # 2. Speed (SI Base: m/s)
    if u in SPEED_TO_MS:
        return "speed", val * SPEED_TO_MS[u]

    # 3. Molar Mass (SI Base: kg/mol)
    if u in MOLAR_MASS_TO_KG_PER_MOL:
        return "molar_mass", val * MOLAR_MASS_TO_KG_PER_MOL[u]

    # 4. Length (SI Base: meters)
    if u in LENGTH_TO_METERS:
        return "length", val * LENGTH_TO_METERS[u]

    # 5. Mass (SI Base: kg)
    if u in MASS_TO_KG:
        return "mass", val * MASS_TO_KG[u]

    # 6. Time (SI Base: seconds)
    if u in TIME_TO_SECONDS:
        return "time", val * TIME_TO_SECONDS[u]

    return None, None


def are_quantities_equivalent(
    val1: float,
    unit1: str,
    val2: float,
    unit2: str,
    rel_tol: Optional[float] = None,
    text_context: str = "",
) -> bool:
    """Check scientific equivalence adhering to Phase 16E domain precision policy."""
    dim1, si1 = to_si_quantity(val1, unit1)
    dim2, si2 = to_si_quantity(val2, unit2)

    if not dim1 or not dim2 or dim1 != dim2 or si1 is None or si2 is None:
        return False

    if abs(si1) < 1e-9 and abs(si2) < 1e-9:
        return True

    if rel_tol is not None:
        denom = max(abs(si1), abs(si2), 1e-9)
        return (abs(si1 - si2) / denom) <= rel_tol

    from app.core.verification.precision_policy import NumericalPrecisionPolicy
    is_equiv, _, _ = NumericalPrecisionPolicy.evaluate_equivalence(
        val1=val1,
        unit1=unit1,
        val2=val2,
        unit2=unit2,
        dimension=dim1,
        text_context=text_context,
    )
    return is_equiv


def evaluate_unit_claim(claim_text: str) -> Optional[Dict[str, Any]]:
    """Evaluates physical unit conversion statements."""
    pattern = re.compile(
        rf"(?i)(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*({RECOGNIZED_UNITS_REGEX})\s*(?:=|equals|is|is equal to)\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*({RECOGNIZED_UNITS_REGEX})"
    )
    m = pattern.search(claim_text.strip())
    if not m:
        return None

    val_from = float(m.group(1))
    unit_from = m.group(2).strip()
    val_to = float(m.group(3))
    unit_to = m.group(4).strip()

    dim_from, si_from = to_si_quantity(val_from, unit_from)
    dim_to, si_to = to_si_quantity(val_to, unit_to)

    if not dim_from or not dim_to or dim_from != dim_to:
        return None

    from app.core.verification.precision_policy import NumericalPrecisionPolicy
    is_consistent, domain, explanation = NumericalPrecisionPolicy.evaluate_equivalence(
        val1=val_from,
        unit1=unit_from,
        val2=val_to,
        unit2=unit_to,
        dimension=dim_from,
        text_context=claim_text,
    )

    return {
        "verified": True,
        "dimension": dim_from,
        "precision_domain": domain.value,
        "val_from": val_from,
        "unit_from": unit_from,
        "claimed_val": val_to,
        "unit_to": unit_to,
        "is_consistent": is_consistent,
        "explanation": explanation,
    }
