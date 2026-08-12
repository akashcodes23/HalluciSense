from types import SimpleNamespace

import pytest

from scripts.run_phase4_gemini_validation import _extract_token_probabilities


def test_extract_token_probabilities_from_attribute_objects():
    candidate = SimpleNamespace(
        logprobs_result=SimpleNamespace(
            chosen_candidates=[
                SimpleNamespace(log_probability=0.0),
                SimpleNamespace(log_probability=-0.69314718056),
            ]
        )
    )

    probs = _extract_token_probabilities(candidate)

    assert probs is not None
    assert probs[0] == pytest.approx(1.0, abs=1e-6)
    assert probs[1] == pytest.approx(0.5, abs=1e-6)


def test_missing_logprobs_is_unavailable_not_zero():
    candidate = SimpleNamespace(logprobs_result=None)
    assert _extract_token_probabilities(candidate) is None


def test_empty_chosen_candidates_is_unavailable():
    candidate = SimpleNamespace(
        logprobs_result=SimpleNamespace(chosen_candidates=[])
    )
    assert _extract_token_probabilities(candidate) is None
