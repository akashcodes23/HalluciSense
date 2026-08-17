"""Deterministic Numerical and Unit Consistency Checker for HalluciSense Enhanced P1.

Extracts numerical quantities, scientific notation, orders of magnitude, and physical/financial units
from claims and evidence snippets. Detects precision errors, scale mismatches, and unit conflicts
without external LLM dependencies.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict


class NumericUnitStatus(str, Enum):
    NO_NUMBERS = "NO_NUMBERS"
    NUMERIC_MATCH = "NUMERIC_MATCH"
    NUMERIC_CONFLICT = "NUMERIC_CONFLICT"
    UNIT_MATCH = "UNIT_MATCH"
    UNIT_CONFLICT = "UNIT_CONFLICT"
    SCALE_CONFLICT = "SCALE_CONFLICT"


# Standard unit synonym map
UNIT_MAP: Dict[str, str] = {
    # Length
    "m": "meter", "metre": "meter", "meters": "meter", "metres": "meter",
    "km": "kilometer", "kilometre": "kilometer", "kilometers": "kilometer", "kilometres": "kilometer",
    "cm": "centimeter", "centimetre": "centimeter", "centimeters": "centimeter", "centimetres": "centimeter",
    "mm": "millimeter", "millimetre": "millimeter", "millimeters": "millimeter", "millimetres": "millimeter",
    "nm": "nanometer", "nanometre": "nanometer", "nanometers": "nanometer", "nanometres": "nanometer",
    "pm": "picometer", "picometre": "picometer", "picometers": "picometer", "picometres": "picometer",
    "μm": "micrometer", "um": "micrometer", "micrometre": "micrometer", "micrometer": "micrometer",
    "micrometres": "micrometer", "micrometers": "micrometer",
    "angstrom": "angstrom", "angstroms": "angstrom", "å": "angstrom",
    "ly": "light_year", "light-year": "light_year", "light-years": "light_year", "light year": "light_year",

    # Mass
    "kg": "kilogram", "kilograms": "kilogram", "kilogramme": "kilogram", "kilogrammes": "kilogram",
    "g": "gram", "grams": "gram", "gramme": "gram", "grammes": "gram",
    "mg": "milligram", "milligrams": "milligram",
    "μg": "microgram", "ug": "microgram", "micrograms": "microgram",
    "ng": "nanogram", "nanograms": "nanogram",
    "tonne": "tonne", "tonnes": "tonne", "tons": "tonne",

    # Time / Speed
    "s": "second", "sec": "second", "seconds": "second",
    "ms": "millisecond", "milliseconds": "millisecond",
    "ns": "nanosecond", "nanoseconds": "nanosecond",
    "min": "minute", "minutes": "minute",
    "h": "hour", "hr": "hour", "hrs": "hour", "hours": "hour",
    "m/s": "m_per_s", "metres per second": "m_per_s", "meters per second": "m_per_s",
    "km/s": "km_per_s", "km/h": "km_per_h",

    # Energy / Power / Force / Constants
    "j": "joule", "joules": "joule", "joule-seconds": "joule_second", "j·s": "joule_second", "j s": "joule_second",
    "kj": "kilojoule", "kilojoules": "kilojoule", "ev": "electronvolt", "electronvolts": "electronvolt",
    "mev": "megaelectronvolt", "gev": "gigaelectronvolt",
    "w": "watt", "watts": "watt", "kw": "kilowatt", "mw": "megawatt",
    "n": "newton", "newtons": "newton", "kn": "kilonewton",
    "m/s²": "m_per_s2", "m/s2": "m_per_s2", "metres per second squared": "m_per_s2",

    # Chemistry / Medicine
    "mol": "mole", "moles": "mole", "mmol": "millimole", "millimoles": "millimole",
    "μmol": "micromole", "umol": "micromole", "micromoles": "micromole",
    "mmol/l": "millimolar", "μmol/l": "micromolar", "umol/l": "micromolar", "mol/l": "molar",
    "kpa": "kilopascal", "pa": "pascal", "mmhg": "mmhg", "atm": "atmosphere",
    "k": "kelvin", "kelvins": "kelvin", "°c": "celsius", "celsius": "celsius", "°f": "fahrenheit",
    "msv": "millisievert", "millisieverts": "millisievert", "sv": "sievert", "sieverts": "sievert",
    "l": "liter", "litre": "liter", "liters": "liter", "litres": "liter",
    "ml": "milliliter", "millilitre": "milliliter", "milliliters": "milliliter", "millilitres": "milliliter",
    "bp": "base_pairs", "base pairs": "base_pairs", "kb": "kilobase", "kilobases": "kilobase", "mb": "megabase",
}

# Metric scale factors relative to base
SCALE_FACTORS: Dict[str, float] = {
    "picometer": 1e-12, "angstrom": 1e-10, "nanometer": 1e-9, "micrometer": 1e-6,
    "millimeter": 1e-3, "centimeter": 1e-2, "meter": 1.0, "kilometer": 1e3,
    "picogram": 1e-12, "nanogram": 1e-9, "microgram": 1e-6, "milligram": 1e-3,
    "gram": 1.0, "kilogram": 1e3, "tonne": 1e6,
    "nanosecond": 1e-9, "millisecond": 1e-3, "second": 1.0, "minute": 60.0, "hour": 3600.0,
    "milliliter": 1e-3, "liter": 1.0,
    "base_pairs": 1.0, "kilobase": 1e3, "megabase": 1e6,
}


@dataclass
class ExtractedQuantity:
    """A numerical value with associated unit and order of magnitude."""
    raw_text: str
    value: float
    exponent: int = 0
    unit: Optional[str] = None
    canonical_unit: Optional[str] = None
    scaled_value: Optional[float] = None


class NumericUnitChecker:
    """Evaluates numerical and unit consistency between claims and reference evidence."""

    def __init__(self, tolerance_pct: float = 0.05):
        self.tolerance_pct = tolerance_pct

    def extract_quantities(self, text: str) -> List[ExtractedQuantity]:
        """Extracts numbers, scientific notations, multiplier words, and units from text."""
        quantities: List[ExtractedQuantity] = []

        # 1. Scientific notation with unicode exponents: e.g., 3×10⁸, 9.11×10⁻³¹, 6.626×10⁻³⁴
        sci_pattern = r"([+-]?\d+(?:\.\d+)?)\s*(?:[×x*]|\\times)?\s*10\s*(?:\^|\*\*)?\s*([+-]?[\d⁻¹²³⁴⁵⁶⁷⁸⁹⁰]+)"
        for match in re.finditer(sci_pattern, text):
            coeff_str, exp_str = match.group(1), match.group(2)
            # convert unicode superscripts
            exp_clean = (exp_str.replace("⁻", "-")
                         .replace("¹", "1").replace("²", "2").replace("³", "3")
                         .replace("⁴", "4").replace("⁵", "5").replace("⁶", "6")
                         .replace("⁷", "7").replace("⁸", "8").replace("⁹", "9")
                         .replace("⁰", "0").replace("+", ""))
            try:
                coeff = float(coeff_str)
                exp = int(exp_clean)
                val = coeff * (10 ** exp)
                # Look for adjacent unit right after
                end_pos = match.end()
                after_text = text[end_pos: end_pos + 35].strip()
                unit = self._find_unit(after_text)
                quantities.append(
                    ExtractedQuantity(
                        raw_text=match.group(0),
                        value=val,
                        exponent=exp,
                        unit=unit,
                        canonical_unit=UNIT_MAP.get(unit.lower(), unit.lower()) if unit else None,
                    )
                )
            except Exception:
                pass

        # 2. Number with word multipliers: e.g., 3 billion, 25 million, 100 thousand
        word_pattern = r"([+-]?\d+(?:\.\d+)?)\s+(billion|million|trillion|thousand|hundred)"
        for match in re.finditer(word_pattern, text, flags=re.IGNORECASE):
            num_str, mult_str = match.group(1), match.group(2).lower()
            mults = {"hundred": 1e2, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
            try:
                val = float(num_str) * mults[mult_str]
                end_pos = match.end()
                after_text = text[end_pos: end_pos + 30].strip()
                unit = self._find_unit(after_text)
                quantities.append(
                    ExtractedQuantity(
                        raw_text=match.group(0),
                        value=val,
                        exponent=int(math.log10(val)) if val > 0 else 0,
                        unit=unit,
                        canonical_unit=UNIT_MAP.get(unit.lower(), unit.lower()) if unit else None,
                    )
                )
            except Exception:
                pass

        # 3. Standard decimal / integer with trailing unit or standalone
        std_pattern = r"(?<!\w)([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z°μ/·\^²³⁻¹]+|\b[a-zA-Z\s]{2,15}\b)?"
        for match in re.finditer(std_pattern, text):
            val_str = match.group(1)
            raw_unit_str = match.group(2) or ""
            # Avoid re-adding numbers matched by scientific / multiplier regex
            if any(q.raw_text in match.group(0) or match.group(0) in q.raw_text for q in quantities):
                continue
            try:
                val = float(val_str)
                unit = self._find_unit(raw_unit_str)
                quantities.append(
                    ExtractedQuantity(
                        raw_text=match.group(0).strip(),
                        value=val,
                        exponent=int(math.log10(abs(val))) if abs(val) >= 1.0 else 0,
                        unit=unit,
                        canonical_unit=UNIT_MAP.get(unit.lower(), unit.lower()) if unit else None,
                    )
                )
            except Exception:
                pass

        return quantities

    def _find_unit(self, text: str) -> Optional[str]:
        """Match potential unit string against UNIT_MAP."""
        if not text:
            return None
        text_clean = text.split(",")[0].split(".")[0].strip().lower()
        # Check longest matching tokens first
        for key in sorted(UNIT_MAP.keys(), key=len, reverse=True):
            if text_clean.startswith(key):
                return key
        return None

    def check_consistency(
        self, claim_text: str, evidence_text: str
    ) -> Tuple[NumericUnitStatus, float, str]:
        """
        Compare numerical values and units in claim vs evidence.
        Returns (Status, Contradiction_Penalty, Explanation).
        """
        claim_quantities = self.extract_quantities(claim_text)
        evidence_quantities = self.extract_quantities(evidence_text)

        if not claim_quantities:
            return NumericUnitStatus.NO_NUMBERS, 0.0, "No numerical quantities found in claim."

        if not evidence_quantities:
            return NumericUnitStatus.NO_NUMBERS, 0.0, "No numerical quantities found in evidence to verify."

        for cq in claim_quantities:
            for eq in evidence_quantities:
                # 1. Check if both have matching or convertible units
                c_unit = cq.canonical_unit
                e_unit = eq.canonical_unit

                if c_unit and e_unit:
                    # Same physical dimension?
                    c_scale = SCALE_FACTORS.get(c_unit)
                    e_scale = SCALE_FACTORS.get(e_unit)

                    if c_scale and e_scale:
                        # Convert to common base
                        c_val_base = cq.value * c_scale
                        e_val_base = eq.value * e_scale

                        ratio = c_val_base / e_val_base if e_val_base != 0 else float("inf")
                        if abs(ratio - 1.0) < self.tolerance_pct:
                            return (
                                NumericUnitStatus.NUMERIC_MATCH,
                                0.0,
                                f"Numerical and unit match (Claim: {cq.value} {cq.unit} ≈ Evidence: {eq.value} {eq.unit})",
                            )
                        elif abs(math.log10(max(1e-12, abs(ratio)))) >= 1.0:
                            # Off by order of magnitude or scale
                            return (
                                NumericUnitStatus.SCALE_CONFLICT,
                                0.90,
                                f"Scale/Unit conflict: claim={cq.value} {cq.unit} vs evidence={eq.value} {eq.unit} (ratio={ratio:.2e})",
                            )
                        else:
                            return (
                                NumericUnitStatus.NUMERIC_CONFLICT,
                                0.85,
                                f"Numerical conflict: claim={cq.value} {cq.unit} vs evidence={eq.value} {eq.unit}",
                            )

                    elif c_unit != e_unit:
                        # Incompatible units
                        return (
                            NumericUnitStatus.UNIT_CONFLICT,
                            0.80,
                            f"Unit conflict: claim uses {cq.unit} but evidence discusses {eq.unit}",
                        )

                # 2. If units are not explicitly tagged, compare raw values / exponents
                if cq.exponent != 0 or eq.exponent != 0:
                    if abs(cq.exponent - eq.exponent) >= 2:
                        return (
                            NumericUnitStatus.SCALE_CONFLICT,
                            0.88,
                            f"Order of magnitude error: claim exponent 10^{cq.exponent} vs evidence 10^{eq.exponent}",
                        )

                # Compare direct numerical values if within comparable range
                if eq.value != 0:
                    ratio = cq.value / eq.value
                    if abs(ratio - 1.0) < self.tolerance_pct:
                        return (
                            NumericUnitStatus.NUMERIC_MATCH,
                            0.0,
                            f"Numerical match: {cq.value} ≈ {eq.value}",
                        )
                    elif ratio > 1.5 or ratio < 0.67:
                        # Significant numerical mismatch
                        return (
                            NumericUnitStatus.NUMERIC_CONFLICT,
                            0.75,
                            f"Numerical value mismatch: claim states {cq.value}, evidence states {eq.value}",
                        )

        return NumericUnitStatus.NO_NUMBERS, 0.0, "No direct numerical conflict detected."
