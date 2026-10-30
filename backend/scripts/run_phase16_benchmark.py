"""Phase 16H — Scientific Factual Regression Benchmark.

Evaluates 12 individual scientific and factual claims through the full
HalluciSense pipeline with Wikipedia retrieval and NLI aggregation.
"""

import json
import os
import sys
import time

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.engine.pipeline import HallucinationDetectionPipeline

BENCHMARK_CLAIMS = [
    {
        "id": 1,
        "text": "The speed of light in vacuum is 299,792,458 m/s.",
        "expected_verdict": "VERIFIED",
        "description": "Defined physical constant (exact SI definition)",
    },
    {
        "id": 2,
        "text": "The molar mass of water is 18.015 g/mol.",
        "expected_verdict": "VERIFIED",
        "description": "Empirical chemical constant within measurement uncertainty",
    },
    {
        "id": 3,
        "text": "Water freezes at 0°C.",
        "expected_verdict": "VERIFIED",
        "description": "Standard state thermodynamic freezing point",
    },
    {
        "id": 4,
        "text": "Water boils at 100°C at standard atmospheric pressure.",
        "expected_verdict": "VERIFIED",
        "description": "Standard state thermodynamic boiling point",
    },
    {
        "id": 5,
        "text": "The speed of light is 150,000 km/s.",
        "expected_verdict": "CONTRADICTED",
        "description": "Factual falsity: 150,000 km/s vs true ~300,000 km/s",
    },
    {
        "id": 6,
        "text": "Bengaluru is the capital of Karnataka.",
        "expected_verdict": "VERIFIED",
        "description": "Geographic capital relation",
    },
    {
        "id": 7,
        "text": "Bengaluru is the capital of Maharashtra.",
        "expected_verdict": "CONTRADICTED",
        "description": "Factual falsity: capital of Maharashtra is Mumbai",
    },
    {
        "id": 8,
        "text": "Water has the chemical formula H2O.",
        "expected_verdict": "VERIFIED",
        "description": "Chemical formula of water",
    },
    {
        "id": 9,
        "text": "Water has the chemical formula CO2.",
        "expected_verdict": "CONTRADICTED",
        "description": "Factual falsity: CO2 is carbon dioxide",
    },
    {
        "id": 10,
        "text": "The Boeing 737 has four engines.",
        "expected_verdict": "CONTRADICTED",
        "description": "Aircraft specification falsity: twin-engine (2) vs 4",
    },
    {
        "id": 11,
        "text": "The Earth orbits the Sun.",
        "expected_verdict": "VERIFIED",
        "description": "Planetary orbital astronomy (heliocentrism)",
    },
    {
        "id": 12,
        "text": "The Earth is the centre of the Solar System.",
        "expected_verdict": "CONTRADICTED",
        "description": "Superseded geocentrism vs true heliocentrism",
    },
]


def run_benchmark():
    pipeline = HallucinationDetectionPipeline()
    results = []

    print("=" * 80)
    print("PHASE 16H — SCIENTIFIC FACTUAL REGRESSION BENCHMARK")
    print("=" * 80)

    for item in BENCHMARK_CLAIMS:
        c_id = item["id"]
        text = item["text"]
        expected = item["expected_verdict"]

        t0 = time.perf_counter()
        report = pipeline.analyze_response(text)
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        p1_res = report.pillar1_summary
        p1_score = p1_res.factual_error_score if p1_res else None
        p1_status = getattr(pipeline.p1_engine, "last_claim_statuses", ["UNKNOWN"])[0] if getattr(pipeline.p1_engine, "last_claim_statuses", None) else "UNKNOWN"
        evidence_list = p1_res.evidence if p1_res else []

        # Find top relevant evidence
        top_ev = None
        for ev in evidence_list:
            if getattr(ev, "relevance_score", 0.0) and getattr(ev, "relevance_score", 0.0) >= 0.40:
                top_ev = ev
                break
        if not top_ev and evidence_list:
            top_ev = evidence_list[0]

        verdict = "VERIFIED" if report.overall_risk_level.value in ("VERIFIED", "LOW_RISK") else (
            "CONTRADICTED" if report.overall_risk_level.value in ("LIKELY_HALLUCINATED", "HALLUCINATED") else report.overall_risk_level.value
        )

        nli_probs = {
            "entailment": getattr(top_ev, "entailment_prob", None),
            "contradiction": getattr(top_ev, "contradiction_prob", None),
            "neutral": getattr(top_ev, "neutral_prob", None),
        } if top_ev else {}

        explanation = report.sentence_analyses[0].reasoning if report.sentence_analyses else (p1_res.reasoning if p1_res else "")

        entry = {
            "claim_id": c_id,
            "claim_text": text,
            "expected_verdict": expected,
            "actual_verdict": verdict,
            "passed": (verdict == expected),
            "p1_factual_error": p1_score,
            "p1_status": p1_status,
            "overall_risk_level": report.overall_risk_level.value,
            "overall_h_score": report.overall_h_score,
            "duration_ms": duration_ms,
            "evidence_count": len(evidence_list),
            "top_evidence_title": getattr(top_ev, "source_name", "None") if top_ev else "None",
            "top_evidence_url": getattr(top_ev, "source_url", "None") if top_ev else "None",
            "top_evidence_snippet": (getattr(top_ev, "snippet", "")[:180] + "...") if top_ev else "",
            "top_evidence_relevance": getattr(top_ev, "relevance_score", None) if top_ev else None,
            "nli_probabilities": nli_probs,
            "explanation": explanation,
        }
        results.append(entry)

        status_flag = "PASS" if entry["passed"] else "FAIL"
        print(f"[{status_flag}] Claim {c_id:2d}: {text}")
        print(f"       Expected: {expected} | Actual: {verdict} | P1 Score: {p1_score} | Top Ev Rel: {entry['top_evidence_relevance']}")
        print(f"       Evidence: {entry['top_evidence_title']} ({entry['top_evidence_url']})")
        print(f"       NLI: {nli_probs}")
        print()

    # Summary
    passed_count = sum(1 for r in results if r["passed"])
    print("=" * 80)
    print(f"BENCHMARK SUMMARY: {passed_count}/{len(results)} CLAIMS PASSED ({(passed_count/len(results))*100:.1f}%)")
    print("=" * 80)

    # Save results to json for reporting
    out_dir = os.path.join(os.path.dirname(__file__), "..", "evaluation_results", "phase16")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase16h_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")
    return results


if __name__ == "__main__":
    run_benchmark()

