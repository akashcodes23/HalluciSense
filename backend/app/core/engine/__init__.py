"""
HalluciSense Engine Package.
Exposes data structures and main detection orchestrator.
"""
from .types import (
    RiskLevel,
    EvidenceItem,
    SentenceAnalysis,
    TokenAnalysis,
    Pillar1Result,
    Pillar2Result,
    Pillar3Result,
    HallucinationReport,
)
from .pipeline import HallucinationDetectionPipeline

__all__ = [
    "RiskLevel",
    "EvidenceItem",
    "SentenceAnalysis",
    "TokenAnalysis",
    "Pillar1Result",
    "Pillar2Result",
    "Pillar3Result",
    "HallucinationReport",
    "HallucinationDetectionPipeline",
]
