"""
HalluciSense Pillar 2 — Multi-LLM Verification Orchestrator
============================================================
Manages parallel verification of claims across multiple LLM verifiers.
"""

import asyncio
import time
from typing import Dict, List, Optional

import structlog
from app.pillar2.multi_llm_verifier.base import BaseLLMVerifier
from app.pillar2.multi_llm_verifier.schemas import (
    MultiLLMVerificationRequest,
    MultiLLMVerificationResponse,
    SingleClaimVerification,
)
from app.pillar2.multi_llm_verifier.verifiers import (
    ClaudeVerifier,
    GeminiVerifier,
    GPTVerifier,
    MockLLMVerifier,
)

logger = structlog.get_logger(__name__)


class MultiLLMVerificationOrchestrator:
    """
    Production orchestrator for Multi-LLM claim verification.
    """

    def __init__(self, verifiers: Optional[List[BaseLLMVerifier]] = None):
        self._verifiers: Dict[str, BaseLLMVerifier] = {}
        if verifiers:
            for v in verifiers:
                self.register_verifier(v)
        else:
            for v in [GeminiVerifier(), GPTVerifier(), ClaudeVerifier(), MockLLMVerifier()]:
                self.register_verifier(v)

    def register_verifier(self, verifier: BaseLLMVerifier) -> None:
        """Register or replace an LLM verifier."""
        self._verifiers[verifier.provider_name.lower()] = verifier
        logger.info("llm_verifier_registered", provider=verifier.provider_name)

    def list_available_verifiers(self) -> List[str]:
        """Return names of registered verifiers."""
        return [v.provider_name for v in self._verifiers.values()]

    async def verify_claim_multi_llm(
        self, request: MultiLLMVerificationRequest
    ) -> MultiLLMVerificationResponse:
        """
        Run multi-LLM verification in parallel.

        Parameters
        ----------
        request : MultiLLMVerificationRequest

        Returns
        -------
        MultiLLMVerificationResponse
        """
        t0 = time.perf_counter()

        if request.verifiers:
            target_keys = [v.lower() for v in request.verifiers]
            active_verifiers = [
                v for key, v in self._verifiers.items() if key in target_keys
            ]
        else:
            active_verifiers = list(self._verifiers.values())

        if not active_verifiers:
            return MultiLLMVerificationResponse(
                claim_id=request.claim_id,
                verifications=[],
                providers_attempted=[],
                failed_verifiers=[],
                total_latency_ms=0.0,
            )

        tasks = [v.verify_claim(request) for v in active_verifiers]
        provider_names = [v.provider_name for v in active_verifiers]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        verifications: List[SingleClaimVerification] = []
        failed: List[str] = []

        for p_name, res in zip(provider_names, results):
            if isinstance(res, Exception) or res is None:
                failed.append(p_name)
                logger.warning("verifier_failed", provider=p_name, error=str(res))
            elif isinstance(res, SingleClaimVerification):
                verifications.append(res)

        total_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "multi_llm_verification_complete",
            claim_id=request.claim_id,
            num_verifications=len(verifications),
            providers_attempted=provider_names,
            total_latency_ms=round(total_ms, 2),
        )

        return MultiLLMVerificationResponse(
            claim_id=request.claim_id,
            verifications=verifications,
            providers_attempted=provider_names,
            failed_verifiers=failed,
            total_latency_ms=round(total_ms, 2),
        )
