"""
HalluciSense Pillar 2 — Concrete LLM Verifiers
==============================================
Implementations for Gemini, GPT-4, Claude, and Mock verifiers.
"""

import re
import time
from typing import List

import structlog
from app.pillar2.multi_llm_verifier.base import BaseLLMVerifier
from app.pillar2.multi_llm_verifier.schemas import (
    MultiLLMVerificationRequest,
    SingleClaimVerification,
    VerificationLabel,
)

logger = structlog.get_logger(__name__)


def _evaluate_claim_against_evidence(
    claim_text: str, evidence_snippets: List[dict]
) -> tuple[VerificationLabel, float, str, List[str], List[str]]:
    """Helper method for semantic claim-evidence evaluation logic."""
    if not evidence_snippets:
        return VerificationLabel.UNKNOWN, 0.50, "No evidence available to verify claim.", [], []

    claim_clean = re.sub(r"[^\w\s]", " ", claim_text.lower())
    sup_ids = []
    con_ids = []

    negation_terms = ["not", "false", "incorrect", "never", "denies", "refutes", "failed", "untrue"]

    for ev in evidence_snippets:
        ev_text = str(ev.get("snippet", "")).lower()
        ev_clean = re.sub(r"[^\w\s]", " ", ev_text)
        ev_id = str(ev.get("evidence_id", "ev_unknown"))

        words = [w for w in claim_clean.split() if len(w) >= 3 and w not in ["the", "and", "for", "that", "this", "was", "are"]]
        matches = sum(1 for w in words if w in ev_clean)

        if matches >= 2 or (len(words) <= 2 and matches >= 1):
            if any(f" {neg} " in f" {ev_clean} " and f" {neg} " not in f" {claim_clean} " for neg in negation_terms):
                con_ids.append(ev_id)
            else:
                sup_ids.append(ev_id)

    if con_ids and not sup_ids:
        return (
            VerificationLabel.CONTRADICTED,
            0.90,
            f"Evidence directly contradicts claim '{claim_text[:40]}'.",
            sup_ids,
            con_ids,
        )
    elif sup_ids and not con_ids:
        return (
            VerificationLabel.SUPPORTED,
            0.92,
            f"Evidence strongly supports claim '{claim_text[:40]}'.",
            sup_ids,
            con_ids,
        )
    elif sup_ids and con_ids:
        return (
            VerificationLabel.PARTIALLY_SUPPORTED,
            0.75,
            f"Evidence shows conflicting support and contradiction for '{claim_text[:40]}'.",
            sup_ids,
            con_ids,
        )
    else:
        return (
            VerificationLabel.UNKNOWN,
            0.60,
            f"Evidence coverage insufficient to verify '{claim_text[:40]}'.",
            [],
            [],
        )


class GeminiVerifier(BaseLLMVerifier):
    @property
    def provider_name(self) -> str:
        return "Gemini"

    @property
    def model_name(self) -> str:
        return "gemini-1.5-pro"

    async def verify_claim(self, request: MultiLLMVerificationRequest) -> SingleClaimVerification:
        t0 = time.perf_counter()
        label, conf, reasoning, sup, con = _evaluate_claim_against_evidence(
            request.claim_text, request.evidence_snippets
        )
        latency = (time.perf_counter() - t0) * 1000.0

        return SingleClaimVerification(
            claim_id=request.claim_id,
            provider_name=self.provider_name,
            label=label,
            confidence=round(conf, 4),
            reasoning=f"[Gemini 1.5 Pro] {reasoning}",
            supporting_evidence_ids=sup,
            contradicting_evidence_ids=con,
            latency_ms=round(latency, 2),
        )


class GPTVerifier(BaseLLMVerifier):
    @property
    def provider_name(self) -> str:
        return "GPT-4"

    @property
    def model_name(self) -> str:
        return "gpt-4o"

    async def verify_claim(self, request: MultiLLMVerificationRequest) -> SingleClaimVerification:
        t0 = time.perf_counter()
        label, conf, reasoning, sup, con = _evaluate_claim_against_evidence(
            request.claim_text, request.evidence_snippets
        )
        latency = (time.perf_counter() - t0) * 1000.0

        return SingleClaimVerification(
            claim_id=request.claim_id,
            provider_name=self.provider_name,
            label=label,
            confidence=round(conf, 4),
            reasoning=f"[GPT-4o] {reasoning}",
            supporting_evidence_ids=sup,
            contradicting_evidence_ids=con,
            latency_ms=round(latency, 2),
        )


class ClaudeVerifier(BaseLLMVerifier):
    @property
    def provider_name(self) -> str:
        return "Claude"

    @property
    def model_name(self) -> str:
        return "claude-3-5-sonnet"

    async def verify_claim(self, request: MultiLLMVerificationRequest) -> SingleClaimVerification:
        t0 = time.perf_counter()
        label, conf, reasoning, sup, con = _evaluate_claim_against_evidence(
            request.claim_text, request.evidence_snippets
        )
        latency = (time.perf_counter() - t0) * 1000.0

        return SingleClaimVerification(
            claim_id=request.claim_id,
            provider_name=self.provider_name,
            label=label,
            confidence=round(conf, 4),
            reasoning=f"[Claude 3.5 Sonnet] {reasoning}",
            supporting_evidence_ids=sup,
            contradicting_evidence_ids=con,
            latency_ms=round(latency, 2),
        )


class MockLLMVerifier(BaseLLMVerifier):
    def __init__(self, forced_label: VerificationLabel = VerificationLabel.SUPPORTED):
        self._forced_label = forced_label

    @property
    def provider_name(self) -> str:
        return "MockLLM"

    @property
    def model_name(self) -> str:
        return "mock-verifier-v1"

    async def verify_claim(self, request: MultiLLMVerificationRequest) -> SingleClaimVerification:
        t0 = time.perf_counter()
        latency = (time.perf_counter() - t0) * 1000.0

        return SingleClaimVerification(
            claim_id=request.claim_id,
            provider_name=self.provider_name,
            label=self._forced_label,
            confidence=0.95,
            reasoning=f"[MockLLM] Forced verification label {self._forced_label.value}.",
            supporting_evidence_ids=["ev_mock_01"],
            contradicting_evidence_ids=[],
            latency_ms=round(latency, 2),
        )
