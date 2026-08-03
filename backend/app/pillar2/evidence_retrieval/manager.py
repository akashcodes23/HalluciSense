"""
HalluciSense Pillar 2 — Evidence Retrieval Manager
===================================================
Orchestrates asynchronous parallel retrieval across multiple evidence providers.
Supports provider selection, timeout controls, fallback handling, and latency benchmarking.
"""

import asyncio
import time
from typing import Dict, List, Optional

import structlog
from app.pillar2.evidence_retrieval.base import BaseEvidenceProvider
from app.pillar2.evidence_retrieval.providers import (
    CrossRefProvider,
    GovDataProvider,
    MockEvidenceProvider,
    PubMedProvider,
    SemanticScholarProvider,
    WikidataProvider,
    WikipediaProvider,
)
from app.pillar2.evidence_retrieval.schemas import EvidenceItem, RetrievalRequest, RetrievalResponse

logger = structlog.get_logger(__name__)


class EvidenceRetrievalManager:
    """
    Production evidence retrieval manager.
    Manages registry of providers and executes concurrent dispatch.
    """

    def __init__(self, providers: Optional[List[BaseEvidenceProvider]] = None):
        self._providers: Dict[str, BaseEvidenceProvider] = {}
        if providers:
            for p in providers:
                self.register_provider(p)
        else:
            # Register default provider stack
            for p in [
                WikipediaProvider(),
                WikidataProvider(),
                CrossRefProvider(),
                SemanticScholarProvider(),
                PubMedProvider(),
                GovDataProvider(),
                MockEvidenceProvider(),
            ]:
                self.register_provider(p)

    def register_provider(self, provider: BaseEvidenceProvider) -> None:
        """Register or replace an evidence provider."""
        self._providers[provider.provider_name.lower()] = provider
        logger.info("evidence_provider_registered", provider=provider.provider_name)

    def list_available_providers(self) -> List[str]:
        """Return names of all registered evidence providers."""
        return [p.provider_name for p in self._providers.values()]

    async def retrieve_evidence(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Concurrently query registered evidence providers.

        Parameters
        ----------
        request : RetrievalRequest

        Returns
        -------
        RetrievalResponse
        """
        t0 = time.perf_counter()
        query = request.query.strip()

        # Select target providers
        if request.providers:
            target_keys = [p.lower() for p in request.providers]
            active_providers = [
                p for key, p in self._providers.items() if key in target_keys
            ]
        else:
            active_providers = list(self._providers.values())

        if not active_providers:
            return RetrievalResponse(
                query=query,
                items=[],
                total_retrieved=0,
                providers_queried=[],
                failed_providers=[],
                total_latency_ms=0.0,
            )

        # Dispatch async tasks with timeouts
        tasks = []
        provider_names = [p.provider_name for p in active_providers]
        for p in active_providers:
            tasks.append(self._retrieve_safe(p, request))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_items: List[EvidenceItem] = []
        failed_providers: List[str] = []

        for p_name, res in zip(provider_names, results):
            if isinstance(res, Exception) or res is None:
                failed_providers.append(p_name)
                logger.warning("evidence_retrieval_failed", provider=p_name, error=str(res))
            elif isinstance(res, list):
                all_items.extend(res)

        total_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "evidence_retrieval_complete",
            query=query[:40],
            total_items=len(all_items),
            providers_queried=provider_names,
            failed_count=len(failed_providers),
            total_latency_ms=round(total_ms, 2),
        )

        return RetrievalResponse(
            query=query,
            items=all_items,
            total_retrieved=len(all_items),
            providers_queried=provider_names,
            failed_providers=failed_providers,
            total_latency_ms=round(total_ms, 2),
        )

    async def _retrieve_safe(
        self, provider: BaseEvidenceProvider, request: RetrievalRequest
    ) -> List[EvidenceItem]:
        """Wrapper ensuring timeout handling."""
        try:
            return await asyncio.wait_for(
                provider.retrieve(request), timeout=request.timeout_seconds
            )
        except Exception as e:
            logger.error("provider_retrieve_error", provider=provider.provider_name, error=str(e))
            raise e
