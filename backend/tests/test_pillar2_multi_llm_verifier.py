"""
Unit tests for HalluciSense Pillar 2 — Module 10.4: Multi-LLM Verification Engine.
"""

import pytest
from app.pillar2.multi_llm_verifier.orchestrator import MultiLLMVerificationOrchestrator
from app.pillar2.multi_llm_verifier.schemas import (
    MultiLLMVerificationRequest,
    VerificationLabel,
)
from app.pillar2.multi_llm_verifier.verifiers import MockLLMVerifier


@pytest.fixture
def orchestrator():
    return MultiLLMVerificationOrchestrator()


def test_list_verifiers(orchestrator):
    verifiers = orchestrator.list_available_verifiers()
    assert "Gemini" in verifiers
    assert "GPT-4" in verifiers
    assert "Claude" in verifiers
    assert "MockLLM" in verifiers


@pytest.mark.asyncio
async def test_multi_llm_verification_supported(orchestrator):
    req = MultiLLMVerificationRequest(
        claim_id="c101",
        claim_text="Einstein published the theory of relativity.",
        evidence_snippets=[
            {
                "evidence_id": "ev_01",
                "snippet": "Einstein published his groundbreaking theory of relativity in physics papers.",
            }
        ],
    )
    res = await orchestrator.verify_claim_multi_llm(req)
    assert res.claim_id == "c101"
    assert len(res.verifications) >= 3
    assert res.total_latency_ms >= 0.0

    labels = [v.label for v in res.verifications]
    assert VerificationLabel.SUPPORTED in labels


@pytest.mark.asyncio
async def test_multi_llm_verification_contradicted(orchestrator):
    req = MultiLLMVerificationRequest(
        claim_id="c102",
        claim_text="The Earth is flat.",
        evidence_snippets=[
            {
                "evidence_id": "ev_02",
                "snippet": "Scientific measurements prove the Earth is not flat, but a spherical oblate spheroid.",
            }
        ],
    )
    res = await orchestrator.verify_claim_multi_llm(req)
    labels = [v.label for v in res.verifications]
    assert VerificationLabel.CONTRADICTED in labels


@pytest.mark.asyncio
async def test_forced_mock_verifier():
    mock_orch = MultiLLMVerificationOrchestrator(
        verifiers=[MockLLMVerifier(forced_label=VerificationLabel.PARTIALLY_SUPPORTED)]
    )
    req = MultiLLMVerificationRequest(
        claim_id="c103",
        claim_text="Sample claim",
        evidence_snippets=[],
    )
    res = await mock_orch.verify_claim_multi_llm(req)
    assert len(res.verifications) == 1
    assert res.verifications[0].label == VerificationLabel.PARTIALLY_SUPPORTED
