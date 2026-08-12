"""Phase 4: Real Gemini + full three-pillar validation.

This runner deliberately exercises the production HalluciSense pipeline with
real Gemini generation metadata instead of substituting values for P2/P3.

Requirements:
    GEMINI_API_KEY must be present in the environment.
    HALLUCISENSE_GEMINI_MODEL must identify a Gemini model that supports
    response log-probabilities for the configured account/API surface.

The controlled benchmark asks Gemini to reproduce each benchmark claim. This
makes token log-probabilities measurable for the exact claim under test while
keeping P1 grounded against the retrieved evidence. Gemini candidates are
also used as alternate generations for P3.

IMPORTANT:
    A successful run must report P2 and P3 as available. If Gemini does not
    return logprobs, the runner fails closed rather than converting missing
    measurements into zero risk.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

from app.core.engine.pipeline import HallucinationDetectionPipeline


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUTPUT_FILE = OUTPUT_DIR / "phase4_gemini_three_pillar_report.json"

CASES: List[Dict[str, str]] = [
    {"category": "TRUE_FACT", "claim": "Apollo 11 landed on the Moon in 1969.", "query": "When and where did Apollo 11 land?"},
    {"category": "FALSE_FACT", "claim": "Apollo 11 landed on Mars in 1969.", "query": "When and where did Apollo 11 land?"},
    {"category": "TEMPORAL_CONTAMINATION", "claim": "The Eiffel Tower was completed in 2020.", "query": "When was the Eiffel Tower completed?"},
    {"category": "ENTITY_SWAP", "claim": "Albert Einstein discovered gravity when an apple fell on his head.", "query": "Who is associated with the falling apple story and who formulated the law of universal gravitation?"},
    {"category": "NUMERICAL_ERROR", "claim": "The distance from Earth to the Moon is 50 miles.", "query": "How far is the Moon from Earth?"},
    {"category": "PARTIAL_TRUTH", "claim": "Neil Armstrong landed on the Moon in 1969 and became President of France in 1975.", "query": "What happened to Neil Armstrong in 1969 and what was his later career?"},
    {"category": "UNVERIFIABLE", "claim": "Quantum computers will definitely achieve sentient consciousness in 2045.", "query": "Will quantum computers definitely achieve sentient consciousness in 2045?"},
    {"category": "CONFLICTING_EVIDENCE", "claim": "Paris is the capital of Japan.", "query": "What is the capital of Japan?"},
    {"category": "MULTI_CLAIM", "claim": "Apollo 11 landed on the Moon in 1969. Neil Armstrong was commander. They landed on Mars in 1975.", "query": "What are the verified facts about Apollo 11 and Neil Armstrong?"},
    {"category": "ADVERSARIAL_CONFIDENCE", "claim": "Without a shadow of a doubt, Abraham Lincoln was elected President of the United States in 2024.", "query": "When was Abraham Lincoln elected President of the United States?"},
]


def _candidate_text(candidate: Any) -> str:
    content = getattr(candidate, "content", None)
    if content is not None:
        parts = getattr(content, "parts", None) or []
        text = "".join(getattr(part, "text", "") or "" for part in parts)
        if text.strip():
            return text.strip()
    return str(getattr(candidate, "text", "") or "").strip()


def _extract_token_probabilities(candidate: Any) -> Optional[List[float]]:
    """Extract chosen-token probabilities from Gemini logprobs output."""
    result = getattr(candidate, "logprobs_result", None)
    if result is None and isinstance(candidate, dict):
        result = candidate.get("logprobs_result")
    if result is None:
        return None

    chosen = getattr(result, "chosen_candidates", None)
    if chosen is None and isinstance(result, dict):
        chosen = result.get("chosen_candidates")
    if not chosen:
        chosen = getattr(result, "chosenCandidates", None)
        if chosen is None and isinstance(result, dict):
            chosen = result.get("chosenCandidates")
    if not chosen:
        return None

    probabilities: List[float] = []
    for item in chosen:
        lp = getattr(item, "log_probability", None)
        if lp is None:
            lp = getattr(item, "logProbability", None)
        if lp is None and isinstance(item, dict):
            lp = item.get("log_probability", item.get("logProbability"))
        if lp is None:
            continue
        try:
            probabilities.append(max(0.0, min(1.0, math.exp(float(lp)))))
        except (TypeError, ValueError, OverflowError):
            continue

    return probabilities or None


def _generate_case(model: Any, claim: str, query: str) -> Tuple[str, Optional[List[float]], List[str], Dict[str, Any]]:
    prompt = f"""You are generating a controlled factual statement for a hallucination-detection experiment.

User question:
{query}

Target statement to reproduce verbatim:
{claim}

Output the target statement exactly and nothing else."""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            candidate_count=3,
            response_logprobs=True,
            logprobs=5,
        ),
        request_options={"timeout": 30},
    )

    candidates = list(getattr(response, "candidates", None) or [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    primary = _candidate_text(candidates[0])
    probabilities = _extract_token_probabilities(candidates[0])
    alternates = []
    for candidate in candidates[1:]:
        text = _candidate_text(candidate)
        if text:
            alternates.append(text)

    metadata = {
        "candidate_count_returned": len(candidates),
        "avg_logprobs": getattr(candidates[0], "avg_logprobs", None),
        "logprobs_available": probabilities is not None,
    }
    return primary, probabilities, alternates, metadata


def run() -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("HALLUCISENSE_GEMINI_MODEL")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required. No Gemini call was attempted.")
    if not model_name:
        raise RuntimeError(
            "HALLUCISENSE_GEMINI_MODEL is required. Choose a Gemini model that "
            "supports response log-probabilities for this account/API surface."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
    pipeline = HallucinationDetectionPipeline()

    results: List[Dict[str, Any]] = []

    for case in CASES:
        started = time.perf_counter()
        primary, token_probs, alternates, generation_meta = _generate_case(
            model,
            case["claim"],
            case["query"],
        )

        if token_probs is None:
            raise RuntimeError(
                f"Gemini did not return token logprobs for {case['category']}. "
                "Pillar 2 cannot be scientifically evaluated from this run."
            )
        if not alternates:
            raise RuntimeError(
                f"Gemini returned no alternate candidate for {case['category']}. "
                "Pillar 3 cannot be scientifically evaluated from this run."
            )

        report = pipeline.analyze(
            text=primary,
            query=case["query"],
            token_probabilities=token_probs,
            sample_responses=alternates,
        )

        p1 = report.pillar1_summary
        p2 = report.pillar2_summary
        p3 = report.pillar3_summary
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not getattr(p2, "available", False):
            raise RuntimeError(f"Pillar 2 unavailable for {case['category']}.")
        if not getattr(p3, "available", False):
            raise RuntimeError(f"Pillar 3 unavailable for {case['category']}.")

        results.append({
            "category": case["category"],
            "target_claim": case["claim"],
            "gemini_primary_response": primary,
            "gemini_alternate_responses": alternates,
            "generation": generation_meta,
            "p1_factual_error": getattr(p1, "factual_error_score", None),
            "p2_confidence_gap": getattr(p2, "confidence_gap_score", None),
            "p2_avg_entropy": getattr(p2, "avg_entropy", None),
            "p3_consistency_failure": getattr(p3, "consistency_failure_score", None),
            "p3_contradiction_score": getattr(p3, "contradiction_score", None),
            "p3_nli_available": getattr(p3, "nli_available", False),
            "h_score": report.overall_h_score,
            "risk_level": getattr(report.overall_risk_level, "value", str(report.overall_risk_level)),
            "weights_used": report.weights_used,
            "elapsed_ms": round(elapsed_ms, 2),
        })

    latencies = [r["elapsed_ms"] for r in results]
    results_sorted = sorted(latencies)

    def percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        k = (len(values) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (k - lo)

    report = {
        "phase": "Phase 4",
        "validation": "real_gemini_full_three_pillar",
        "model": model_name,
        "cases": len(results),
        "strict_availability": {
            "p2_available_for_all": all(r["p2_confidence_gap"] is not None for r in results),
            "p3_available_for_all": all(r["p3_consistency_failure"] is not None for r in results),
            "p3_nli_available_for_all": all(r["p3_nli_available"] for r in results),
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(percentile(results_sorted, 0.50), 2),
            "p95": round(percentile(results_sorted, 0.95), 2),
            "p99": round(percentile(results_sorted, 0.99), 2),
        },
        "results": results,
        "scientific_note": (
            "This is a controlled Gemini generation benchmark. The target claim is explicitly requested "
            "so token logprobs can be measured for the claim under test. It validates end-to-end availability "
            "and fusion of P1/P2/P3; it is not a substitute for evaluation on naturally generated LLM outputs."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved: {OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    run()
