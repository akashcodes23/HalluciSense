"""
HalluciSense Pillar 2 — Abstract Multi-LLM Verifier Interface
==============================================================
Contract for LLM verification providers (Gemini, GPT, Claude, Mock).
"""

from abc import ABC, abstractmethod
from app.pillar2.multi_llm_verifier.schemas import MultiLLMVerificationRequest, SingleClaimVerification


class BaseLLMVerifier(ABC):
    """
    Abstract Base Class for Multi-LLM Claim Verifiers.
    Normalizes provider outputs into standardized VerificationLabels.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of LLM provider (e.g., 'Gemini', 'GPT-4', 'Claude-3.5')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Specific model identifier."""
        pass

    @abstractmethod
    async def verify_claim(self, request: MultiLLMVerificationRequest) -> SingleClaimVerification:
        """
        Execute asynchronous claim verification against evidence.

        Parameters
        ----------
        request : MultiLLMVerificationRequest

        Returns
        -------
        SingleClaimVerification
        """
        pass
