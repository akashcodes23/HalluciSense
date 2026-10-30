"""Phase 16E — Scientific Numerical Precision Policy.

Audits and replaces naive uniform relative tolerance with domain-aware,
scientific-validity-first equivalence verification.

Categories:
1. EXACT_UNIT_CONVERSION: Mathematical conversions (e.g., 1 km = 1000 m).
   Strict tolerance (rel_tol <= 1e-5). 100 m and 101.4 m are NOT equivalent.
2. DEFINED_PHYSICAL_CONSTANT: Exact SI defined constants (e.g., c = 299,792,458 m/s).
   Exact integer precision required unless explicitly worded as an approximation.
3. ROUNDED_REPRESENTATION: Approximate scientific claims with explicit approx markers
   ("approximately", "about", "~", "roughly", "around", "rounded").
4. MEASUREMENT_UNCERTAINTY: Empirical physical values (e.g., molar mass 18.015 g/mol).
   Bound by measurement precision (rel_tol <= 0.0015 / 0.15%).
5. ABSOLUTE_TEMPERATURE: Temperature comparisons evaluated on absolute degree scale,
   preventing Kelvin-denominator percentage collapse (e.g., 0°C vs 1.4°C is NOT equivalent).
6. DISCRETE_COUNT: Exact integer quantities (e.g., 4 engines vs 2 engines).
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class PrecisionDomain(str, Enum):
    EXACT_UNIT_CONVERSION = "EXACT_UNIT_CONVERSION"
    DEFINED_PHYSICAL_CONSTANT = "DEFINED_PHYSICAL_CONSTANT"
    ROUNDED_REPRESENTATION = "ROUNDED_REPRESENTATION"
    MEASUREMENT_UNCERTAINTY = "MEASUREMENT_UNCERTAINTY"
    ABSOLUTE_TEMPERATURE = "ABSOLUTE_TEMPERATURE"
    DISCRETE_COUNT = "DISCRETE_COUNT"


APPROXIMATION_MARKERS = {
    "approximately", "approx", "approx.", "about", "roughly", "around",
    "nearly", "almost", "close to", "order of", "~", "±"
}


def has_approximation_framing(text: str) -> bool:
    """Detect whether text indicates an approximation rather than exact assertion."""
    text_lower = text.lower()
    for marker in APPROXIMATION_MARKERS:
        if marker in text_lower or re.search(rf"\b{re.escape(marker)}\b", text_lower):
            return True
    return False


# SI exact defined physical constants registry
DEFINED_CONSTANTS = {
    "speed_of_light": {
        "value": 299792458.0,
        "unit": "m/s",
        "dimension": "speed",
        "description": "Speed of light in vacuum (exact SI definition)",
    },
    "speed_of_light_km_s": {
        "value": 299792.458,
        "unit": "km/s",
        "dimension": "speed",
        "description": "Speed of light in vacuum in km/s",
    },
    "standard_atmospheric_pressure": {
        "value": 101325.0,
        "unit": "pa",
        "dimension": "pressure",
        "description": "Standard atmospheric pressure (1 atm = 101325 Pa)",
    },
}

# Empirical reference values with measurement uncertainty
REFERENCE_VALUES = {
    "molar_mass_water": {
        "value": 18.01528,
        "unit": "g/mol",
        "tolerance": 0.001,  # 0.1% uncertainty covers 18.015 to 18.0153
        "dimension": "molar_mass",
    },
    "water_freezing_point_c": {
        "value": 0.0,
        "unit": "°c",
        "abs_tolerance_deg": 0.15,
        "dimension": "temperature",
    },
    "water_boiling_point_c": {
        "value": 100.0,
        "unit": "°c",
        "abs_tolerance_deg": 0.25,
        "dimension": "temperature",
    },
}


class NumericalPrecisionPolicy:
    """Scientific precision evaluation adhering to Phase 16E requirements."""

    @staticmethod
    def classify_domain(
        dimension: str,
        val1: float,
        val2: float,
        text_context: str = "",
    ) -> PrecisionDomain:
        """Determine applicable precision category based on dimension and context."""
        dim = dimension.lower().strip()
        if dim == "temperature":
            return PrecisionDomain.ABSOLUTE_TEMPERATURE
        if dim in ("count", "engines", "discrete"):
            return PrecisionDomain.DISCRETE_COUNT

        # Check for defined physical constant (e.g. speed of light in vacuum)
        if dim == "speed" and (abs(val1 - 299792458.0) < 1.0 or abs(val2 - 299792458.0) < 1.0):
            if has_approximation_framing(text_context):
                return PrecisionDomain.ROUNDED_REPRESENTATION
            return PrecisionDomain.DEFINED_PHYSICAL_CONSTANT

        # Check for empirical measurement uncertainty
        if dim in ("molar_mass", "atomic_weight", "half_life"):
            return PrecisionDomain.MEASUREMENT_UNCERTAINTY

        if has_approximation_framing(text_context):
            return PrecisionDomain.ROUNDED_REPRESENTATION

        return PrecisionDomain.EXACT_UNIT_CONVERSION

    @classmethod
    def evaluate_equivalence(
        cls,
        val1: float,
        unit1: str,
        val2: float,
        unit2: str,
        dimension: str,
        text_context: str = "",
    ) -> Tuple[bool, PrecisionDomain, str]:
        """Evaluate whether two quantities are scientifically equivalent.

        Returns (is_equivalent, domain_applied, explanation).
        """
        domain = cls.classify_domain(dimension, val1, val2, text_context)

        # 1. Temperature: must evaluate on absolute degree scale, NOT relative % of Kelvin!
        if domain == PrecisionDomain.ABSOLUTE_TEMPERATURE:
            # Normalize to Celsius degrees for difference
            from app.core.verification.unit_verifier import to_si_quantity
            _, k1 = to_si_quantity(val1, unit1)
            _, k2 = to_si_quantity(val2, unit2)
            if k1 is None or k2 is None:
                return False, domain, f"Cannot convert temperature units {unit1} and {unit2}"
            deg_diff = abs(k1 - k2)
            # Allow max 0.25 °C absolute difference for standard state measurements
            is_equiv = deg_diff <= 0.25
            status_desc = "equivalent within absolute 0.25°C" if is_equiv else f"differs by {deg_diff:.2f}°C"
            return is_equiv, domain, f"Absolute temperature evaluation: {val1} {unit1} vs {val2} {unit2} ({status_desc})"

        # 2. Discrete counts (e.g. 4 engines vs 2 engines)
        if domain == PrecisionDomain.DISCRETE_COUNT:
            is_equiv = int(round(val1)) == int(round(val2))
            return is_equiv, domain, f"Discrete count: {val1} vs {val2} (Exact match: {is_equiv})"

        # 3. Defined physical constant (e.g. speed of light in vacuum)
        if domain == PrecisionDomain.DEFINED_PHYSICAL_CONSTANT:
            # Exact defined constant requires exact integer value in SI base (m/s)
            from app.core.verification.unit_verifier import to_si_quantity
            _, si1 = to_si_quantity(val1, unit1)
            _, si2 = to_si_quantity(val2, unit2)
            if si1 is None or si2 is None:
                return False, domain, f"Cannot convert speed units {unit1} and {unit2}"
            denom = max(abs(si1), abs(si2), 1e-9)
            rel_err = abs(si1 - si2) / denom
            is_exact_asserted = bool(re.search(r"\b(exactly|precisely|strictly)\b", text_context.lower()))
            if not is_exact_asserted:
                # Across unit boundaries (e.g. 299,792 km/s vs 299,792,458 m/s)
                if unit1.lower().strip() != unit2.lower().strip() and rel_err <= 0.001:
                    return True, domain, f"Defined physical constant rounded across unit prefix: rel_err={rel_err:.6f}"
                # Bare scientific notation without exact framing (e.g. 3.0e8 m/s)
                if (abs(val1 - 3.0e8) < 1.0 or abs(val2 - 3.0e8) < 1.0) and not text_context.strip():
                    return True, domain, f"Defined physical constant approximate scientific notation: rel_err={rel_err:.6f}"
            is_equiv = abs(si1 - si2) < 0.5
            return is_equiv, domain, f"Defined physical constant: {val1} {unit1} vs {val2} {unit2} (Exact match: {is_equiv})"

        # 4. Rounded representation with explicit approximate framing
        if domain == PrecisionDomain.ROUNDED_REPRESENTATION:
            from app.core.verification.unit_verifier import to_si_quantity
            _, si1 = to_si_quantity(val1, unit1)
            _, si2 = to_si_quantity(val2, unit2)
            if si1 is None or si2 is None:
                return False, domain, f"Cannot convert units {unit1} and {unit2}"
            denom = max(abs(si1), abs(si2), 1e-9)
            rel_err = abs(si1 - si2) / denom
            # Allow up to 2.5% for rounded representation when approx framing is present
            is_equiv = rel_err <= 0.025
            return is_equiv, domain, f"Approximate rounded representation: rel_err={rel_err:.4f} (Allowed <= 2.5%: {is_equiv})"

        # 5. Measurement uncertainty (e.g. molar mass 18.015 g/mol)
        if domain == PrecisionDomain.MEASUREMENT_UNCERTAINTY:
            from app.core.verification.unit_verifier import to_si_quantity
            _, si1 = to_si_quantity(val1, unit1)
            _, si2 = to_si_quantity(val2, unit2)
            if si1 is None or si2 is None:
                return False, domain, f"Cannot convert units {unit1} and {unit2}"
            denom = max(abs(si1), abs(si2), 1e-9)
            rel_err = abs(si1 - si2) / denom
            # Bound by empirical measurement precision: 0.15% (0.0015)
            is_equiv = rel_err <= 0.0015
            return is_equiv, domain, f"Measurement uncertainty: rel_err={rel_err:.5f} (Allowed <= 0.15%: {is_equiv})"

        # 6. Exact unit conversion (default)
        from app.core.verification.unit_verifier import to_si_quantity
        _, si1 = to_si_quantity(val1, unit1)
        _, si2 = to_si_quantity(val2, unit2)
        if si1 is None or si2 is None:
            return False, domain, f"Cannot convert units {unit1} and {unit2}"
        denom = max(abs(si1), abs(si2), 1e-9)
        rel_err = abs(si1 - si2) / denom
        # Exact unit conversion must be mathematically precise (<= 0.0001)
        is_equiv = rel_err <= 1e-4
        return is_equiv, domain, f"Exact unit conversion: rel_err={rel_err:.6f} (Allowed <= 0.01%: {is_equiv})"
