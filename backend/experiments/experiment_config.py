"""Phase 21 — Experiment Configuration Schema.

Validates experiment parameters, dataset choices, model choices, fusion weights,
and threshold choices using Pydantic.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    name: str = Field(..., description="Experiment human-readable title")
    benchmark_dataset: str = Field("TruthfulQA", description="Dataset identifier")
    model_name: str = Field("GPT-4", description="Target LLM model name")
    sample_count: int = Field(100, description="Evaluation sample size N")
    random_seed: int = Field(42, description="Fixed random seed S")
    fusion_mode: str = Field("ADAPTIVE", description="Fusion mode: STATIC, ADAPTIVE, GRADIENT")
    threshold: float = Field(0.54, description="Decision risk threshold tau*")
    platt_params: List[float] = Field(default_factory=lambda: [1.82, -0.45], description="Platt scale [a, b]")
    enable_knowledge_graph: bool = Field(True, description="Enable Pillar 3 Knowledge Graph")
    enable_token_localization: bool = Field(True, description="Enable 4-tier token localization")
