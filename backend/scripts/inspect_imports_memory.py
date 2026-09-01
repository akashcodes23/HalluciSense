"""Phase 49 Step 1 & 2: Complete Allocation Graph & Module-by-Module Memory Profiler."""

import gc
import os
import sys
import psutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

process = psutil.Process(os.getpid())

def get_rss() -> float:
    return process.memory_info().rss / (1024.0 * 1024.0)

def profile_imports():
    print("=" * 80)
    print("PHASE 49: MODULE-BY-MODULE IMPORT MEMORY ALLOCATION MAPPING")
    print("=" * 80)
    
    modules_to_test = [
        ("Base Python Runtime", None),
        ("numpy", "import numpy as np"),
        ("scipy", "import scipy"),
        ("sklearn", "import sklearn"),
        ("joblib", "import joblib"),
        ("torch", "import torch"),
        ("torchvision", "import torchvision"),
        ("transformers", "from transformers import AutoTokenizer, AutoModelForSequenceClassification"),
        ("sentence_transformers", "import sentence_transformers"),
        ("fastapi", "import fastapi"),
        ("httpx", "import httpx"),
        ("structlog", "import structlog"),
        ("pydantic", "import pydantic"),
        ("app.core.config", "from app.core.config import settings"),
        ("app.core.engine.model_registry", "from app.core.engine.model_registry import ModelRegistry"),
        ("app.core.engine.pillar1_retrieval", "from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine"),
        ("app.core.engine.pillar2_confidence", "from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine"),
        ("app.core.engine.pillar3_consistency", "from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine"),
        ("app.core.engine.fusion", "from app.core.engine.fusion import AdaptiveFusionEngine"),
        ("app.core.engine.pipeline", "from app.core.engine.pipeline import HallucinationDetectionPipeline"),
        ("app.main", "from app.main import create_application"),
    ]

    r_prev = get_rss()
    print(f"Initial RSS: {r_prev:.2f} MB\n")
    print(f"{'Module / Subsystem':<42} | {'Current RSS':>12} | {'Delta':>10}")
    print("-" * 70)

    for name, imp_stmt in modules_to_test:
        if imp_stmt:
            exec(imp_stmt)
        r_curr = get_rss()
        delta = r_curr - r_prev
        sign = "+" if delta >= 0 else ""
        print(f"{name:<42} | {r_curr:9.2f} MB | {sign}{delta:7.2f} MB")
        r_prev = r_curr

if __name__ == "__main__":
    profile_imports()
