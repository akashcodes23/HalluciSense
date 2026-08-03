"""
HalluciSense Pillar 2 — Abstract Evidence Provider Interface
============================================================
Defines the contract for all domain-specific evidence retrieval providers.
"""

from abc import ABC, abstractmethod
from typing import List

from app.pillar2.evidence_retrieval.schemas import EvidenceItem, RetrievalRequest


class BaseEvidenceProvider(ABC):
    """
    Abstract Base Class for Evidence Retrieval Providers.
    Ensures provider independence, zero hardcoded couplings, and modularity.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique string name of provider (e.g. 'Wikipedia', 'PubMed')."""
        pass

    @property
    @abstractmethod
    def default_authority_score(self) -> float:
        """Default domain authority score between 0.0 and 1.0."""
        pass

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> List[EvidenceItem]:
        """
        Execute asynchronous evidence retrieval for query.

        Parameters
        ----------
        request : RetrievalRequest

        Returns
        -------
        List[EvidenceItem]
        """
        pass
