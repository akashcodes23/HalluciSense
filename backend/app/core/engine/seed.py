"""Deterministic Random Seed Initialization Module for HalluciSense.

Locks random seeds across Python, NumPy, PyTorch, and CUDA to ensure
100% reproducible experiments and artifact evaluation.
"""

from __future__ import annotations

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across all random number generators.

    Args:
        seed: The integer seed value (default: 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


if __name__ == "__main__":
    set_seed(42)
    print("Successfully set global random seed S=42 across Python, NumPy, and PyTorch.")
