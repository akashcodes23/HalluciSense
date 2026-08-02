"""Phase 6J — Shared utility functions for numerical analysis.

Provides common JSON serialization helpers for converting NumPy data structures
and non-finite floating point values into native, JSON-compliant Python types.

This module is analysis-only and read-only.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert NumPy data types and non-finite floats to native Python types.

    Handles nested dictionaries, lists, tuples, NumPy scalar types, and NumPy ndarrays.
    NaN and ±Inf float values are converted to ``None`` to ensure valid JSON output.

    Args:
        obj: Any Python/NumPy data structure or scalar.

    Returns:
        JSON-serializable Python native structure.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.ndarray):
        return None
    return obj


# Alias for backward compatibility within Phase 6J modules
_serializable = make_json_serializable
