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


def __getattr__(name: str):
    if name == "HallucinationDetectionPipeline":
        from .pipeline import HallucinationDetectionPipeline
        return HallucinationDetectionPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
