"""HalluciSense v1.0 Canonical Production Router.

Implements production REST API endpoints:
- POST /api/v1/analyze
- POST /api/v1/explain
- GET  /api/v1/debug/latest
- GET  /api/v1/debug/{trace_id}
- GET  /api/v1/metrics
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, Request, Response, status
from app.schemas.production_schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ExplainRequest,
    ExplainResponse,
    MetricsResponse,
    PillarScores,
    SentenceScore,
    TokenHeatmapItem,
    EvidenceItem,
    ConfidenceAnalysis,
    MeasuredTimingBreakdown,
    PillarExecutionStatus,
    MathematicalFusionDecomposition,
)
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.token_localization import TokenLevelLocalizationEngine
from app.core.engine.tracer import PipelineTracer, get_latest_trace, get_trace_by_id
from app.core.engine.root_cause_classifier import RootCauseClassifier
from app.core.engine.metrics_tracker import get_metrics_tracker

router = APIRouter(tags=["Analysis"])

# Lazy singletons for production engines via ModelRegistry
from app.core.engine.model_registry import ModelRegistry
_localization_engine = TokenLevelLocalizationEngine()
_metrics_tracker = get_metrics_tracker()

def get_pipeline():
    return ModelRegistry.get_pipeline()

MAX_PAYLOAD_BYTES = 100 * 1024  # 100 KB limit

VALID_MODELS = {
    "gpt-4", "gpt-4.1", "gemini", "claude", "llama-3", "qwen", "mistral", "deepseek", "phi",
    "gpt-4o", "gpt-3.5-turbo", "claude-3-5-sonnet", "llama-3-70b", "default"
}


from app.core.engine.pipeline_timer import PipelineTimer, StageMeasurement

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Canonical production hallucination analysis",
    description="Executes Pillar 1 Retrieval, Pillar 2 Confidence, Pillar 3 Consistency, Adaptive Fusion, Token Localization, and single-label failure classification.",
)
async def analyze_response(payload: AnalysisRequest, request: Request, response: Response) -> AnalysisResponse:
    """Execute end-to-end HalluciSense verification pipeline with full tracing."""
    tracer = PipelineTracer()
    timer = PipelineTimer(trace_id=tracer.trace_id)

    # 0. Readiness Gate Check
    readiness = getattr(request.app.state, "readiness_status", "READY")
    if readiness != "READY":
        _metrics_tracker.record_request(0.0, 0.0, is_success=False)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_NOT_READY",
                "message": "HalluciSense verification pipeline is still initializing.",
            },
        )

    # 0.1 Resource Pressure Guard (OOM Prevention)
    try:
        import os, psutil
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        if rss_mb > 1750.0:
            _metrics_tracker.record_request(0.0, 0.0, is_success=False)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "RESOURCE_PRESSURE",
                    "message": "Verification capacity is temporarily saturated due to high memory pressure. Please retry shortly.",
                },
            )
    except HTTPException:
        raise
    except Exception:
        pass

    # 1. Request Initialization
    with timer.stage("request_initialization"):
        body_bytes = await request.body()

    # Payload size check
    if len(body_bytes) > MAX_PAYLOAD_BYTES:
        _metrics_tracker.record_request(0.0, 0.0, is_success=False)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request payload size ({len(body_bytes)} bytes) exceeds maximum limit of {MAX_PAYLOAD_BYTES} bytes.",
        )

    # 2. Input Validation
    with timer.stage("input_validation"):
        query = payload.query.strip() if (payload.query and payload.query.strip()) else None
        response_text = payload.response.strip() if payload.response else ""
        model_name = (payload.model_name or "gpt-4").strip().lower()

        if not response_text:
            _metrics_tracker.record_request(0.0, 0.0, is_success=False)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'response' must contain non-empty, non-whitespace string values.",
            )

        if model_name not in VALID_MODELS and not any(m in model_name for m in ["gpt", "gemini", "claude", "llama", "qwen", "mistral"]):
            _metrics_tracker.record_request(0.0, 0.0, is_success=False)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported model_name '{payload.model_name}'. Supported options include: {sorted(list(VALID_MODELS))}.",
            )

    try:
        # 3. Master Pipeline Execution (Async Offloaded with real payload arguments)
        ev_items = None
        if payload.provided_evidence:
            from app.core.engine.types import EvidenceItem as CoreEvidenceItem
            ev_items = []
            for ev in payload.provided_evidence:
                if isinstance(ev, str) and ev.strip():
                    ev_items.append(
                        CoreEvidenceItem(
                            claim=query or response_text[:100],
                            snippet=ev.strip(),
                            source_name="Provided Evidence Context",
                            similarity_score=1.0,
                        )
                    )
                elif isinstance(ev, dict) and ev.get("snippet"):
                    ev_items.append(
                        CoreEvidenceItem(
                            claim=ev.get("claim", query or response_text[:100]),
                            snippet=ev["snippet"].strip(),
                            source_name=ev.get("source_name", "Provided Evidence"),
                            source_url=ev.get("source_url"),
                            similarity_score=float(ev.get("similarity_score", ev.get("score", 1.0))),
                        )
                    )

        t0 = time.perf_counter()
        report = await asyncio.to_thread(
            get_pipeline().analyze,
            text=response_text,
            query=query,
            token_probabilities=payload.logprobs,
            sample_responses=payload.sample_responses,
            evidence_items=ev_items,
        )
        p_dur = (time.perf_counter() - t0) * 1000.0

        overall_h = float(report.overall_h_score)
        risk_level_str = str(report.overall_risk_level.value) if hasattr(report.overall_risk_level, "value") else str(report.overall_risk_level)

        # Extract Pillar Scores & Measured Timings from Report
        p1 = report.pillar1_summary
        p2 = report.pillar2_summary
        p3 = report.pillar3_summary
        p_timings = getattr(report, "performance_timings", {}) or {}

        # Measured sub-operation durations from perf_counter
        ret_perf = p_timings.get("retrieval", {})
        conf_perf = p_timings.get("confidence", {})
        cons_perf = p_timings.get("consistency", {})
        fus_perf = p_timings.get("fusion", {})

        p1_dur_ms = float(ret_perf.get("duration_ms", 0.0))
        p2_dur_ms = float(conf_perf.get("duration_ms", 0.0))
        p3_dur_ms = float(cons_perf.get("duration_ms", 0.0))
        fusion_dur_ms = float(fus_perf.get("duration_ms", 0.0))

        timer.record("query_preprocessing", ret_perf.get("claim_extraction_ms", 0.0))
        timer.record("sentence_segmentation", 0.0)
        timer.record("retrieval", p1_dur_ms)
        timer.record("retrieval_bm25", ret_perf.get("bm25_ms", 0.0))
        timer.record("retrieval_dense", ret_perf.get("wikipedia_ms", 0.0) + ret_perf.get("faiss_ms", 0.0))
        timer.record("retrieval_hybrid_fusion", ret_perf.get("reranker_ms", 0.0))
        timer.record("nli_verification", ret_perf.get("nli_ms", 0.0) + cons_perf.get("nli_ms", 0.0))
        timer.record("confidence_estimation", p2_dur_ms)
        timer.record("consistency_reasoning", p3_dur_ms)
        timer.record("consistency_paraphrase", cons_perf.get("sanitization_ms", 0.0))
        timer.record("consistency_multi_run", 0.0)
        timer.record("consistency_comparison", cons_perf.get("semantic_ms", 0.0) + cons_perf.get("nli_ms", 0.0))
        timer.record("adaptive_fusion", fusion_dur_ms)
        timer.record("calibration", 0.0)

        p1_available = True
        p2_available = bool(p2 and getattr(p2, "available", False) and getattr(p2, "confidence_gap_score", None) is not None)
        p3_available = bool(p3 and getattr(p3, "available", False) and getattr(p3, "consistency_failure_score", None) is not None)
        is_full_analysis = bool(p1_available and p2_available and p3_available)

        fe_val = float(getattr(p1, "factual_error_score", 0.0))
        cg_val = float(p2.confidence_gap_score) if p2_available else 0.0
        cf_val = float(p3.consistency_failure_score) if p3_available else 0.0

        pillar_scores = PillarScores(
            retrieval=round(fe_val, 4),
            confidence=round(cg_val, 4) if p2_available else None,
            consistency=round(cf_val, 4) if p3_available else None,
        )

        p_status_obj = PillarExecutionStatus(
            p1_status="EXECUTED" if p1 else "FAILED",
            p2_status="EXECUTED" if p2_available else "UNAVAILABLE",
            p3_status="EXECUTED" if p3_available else "UNAVAILABLE",
            fusion_status="FULL_THREE_PILLAR" if is_full_analysis else ("PARTIAL_TWO_PILLAR" if (p2_available or p3_available) else "PARTIAL_ONE_PILLAR"),
            p1_available=p1_available,
            p2_available=p2_available,
            p3_available=p3_available,
            is_full_analysis=is_full_analysis,
        )

        # Mathematical fusion breakdown
        weights_used = getattr(report, "weights_used", {}) or {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}
        w_alpha = float(weights_used.get("alpha", weights_used.get("alpha_factual_error", 0.45)))
        w_beta = float(weights_used.get("beta", weights_used.get("beta_confidence_gap", 0.30)))
        w_gamma = float(weights_used.get("gamma", weights_used.get("gamma_consistency_failure", 0.25)))

        c_p1 = round(w_alpha * fe_val, 4)
        c_p2 = round(w_beta * cg_val, 4) if p2_available else None
        c_p3 = round(w_gamma * cf_val, 4) if p3_available else None

        calibrated_h = float(getattr(report, "calibrated_probability", overall_h) or overall_h)

        avail_pillars = []
        if p1_available:
            avail_pillars.append("Pillar 1: Evidence Grounding")
        if p2_available:
            avail_pillars.append("Pillar 2: Confidence Estimation")
        if p3_available:
            avail_pillars.append("Pillar 3: Consistency Reasoning")

        missing_pillars = []
        if not p2_available:
            missing_pillars.append("Pillar 2: Confidence (Token logprobs omitted)")
        if not p3_available:
            missing_pillars.append("Pillar 3: Consistency (Single generation mode)")

        fusion_mode_str = "FULL_THREE_PILLAR" if is_full_analysis else "PARTIAL_RENORMALIZED"

        if is_full_analysis:
            decomp_explanation = (
                f"Full 3-Pillar weighted fusion: H = 0.45*P1 + 0.30*P2 + 0.25*P3 = "
                f"{c_p1:.4f} + {c_p2 or 0.0:.4f} + {c_p3 or 0.0:.4f} = {overall_h:.4f} ({overall_h * 100:.2f}%)."
            )
        else:
            missing_desc = "; ".join(missing_pillars)
            decomp_explanation = (
                f"Partial renormalized fusion: {missing_desc}. Score computed from available pillars with effective weights "
                f"(α={w_alpha:.2f}, β={w_beta:.2f}, γ={w_gamma:.2f}) yielding H = {overall_h:.4f} ({overall_h * 100:.2f}%)."
            )

        fusion_decomp = MathematicalFusionDecomposition(
            equation="H = alpha*P1 + beta*P2 + gamma*P3",
            fusion_mode=fusion_mode_str,
            configured_weights={"alpha": 0.45, "beta": 0.30, "gamma": 0.25},
            effective_weights={"alpha": w_alpha, "beta": w_beta, "gamma": w_gamma},
            pillar_scores={
                "p1_factual_error": fe_val,
                "p2_confidence_gap": cg_val if p2_available else None,
                "p3_consistency_failure": cf_val if p3_available else None,
            },
            weighted_contributions={
                "p1_contribution": c_p1,
                "p2_contribution": c_p2,
                "p3_contribution": c_p3,
            },
            available_pillars=avail_pillars,
            missing_pillars=missing_pillars,
            uncalibrated_h_score=round(overall_h, 4),
            calibrated_h_score=round(calibrated_h, 4),
            is_full_analysis=is_full_analysis,
            explanation=decomp_explanation,
        )

        # 4. Token Localization Heatmap (Async Offloaded)
        with timer.stage("token_localization"):
            annotations, _ = await asyncio.to_thread(
                _localization_engine.localize_tokens,
                response_text=response_text,
                overall_h_score=overall_h,
                sentence_scores=[overall_h],
            )
        loc_m = timer.measurements.get("token_localization")
        localization_ms = loc_m.duration_ms if loc_m else 0.0

        heatmap_items = []
        for ann in annotations:
            heatmap_items.append(
                TokenHeatmapItem(
                    token=ann["text"],
                    score=ann["risk_score"],
                    tier=ann["risk_tier"],
                    color_hex=ann["color_hex"],
                )
            )

        # 5. Sentence Scores
        sentences = []
        for s in report.sentence_analyses:
            s_risk = str(s.risk_level.value) if hasattr(s.risk_level, "value") else str(s.risk_level)
            sentences.append(
                SentenceScore(
                    sentence_index=s.sentence_id,
                    text=s.text,
                    score=round(float(s.hallucination_score), 4),
                    risk_level=s_risk,
                )
            )

        if not sentences:
            sentences.append(
                SentenceScore(
                    sentence_index=0,
                    text=response_text,
                    score=round(overall_h, 4),
                    risk_level=risk_level_str,
                )
            )

        # 6. Real Evidence Citations (Zero Placeholders)
        evidence_items = []
        for idx, item in enumerate(p1.evidence):
            p_dict = item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else (item.dict() if hasattr(item, "dict") else {}))
            snippet_str = str(p_dict.get("snippet", p_dict.get("text", ""))).strip()
            if not snippet_str:
                continue
            evidence_items.append(
                EvidenceItem(
                    id=str(p_dict.get("id", f"ev_{idx+1}")),
                    title=str(p_dict.get("title", p_dict.get("source_name", "Retrieved Evidence"))),
                    snippet=snippet_str,
                    score=float(p_dict.get("similarity_score", p_dict.get("score", 0.5))),
                    source=str(p_dict.get("source_name", "Wikipedia / BM25+Dense Index")),
                )
            )

        # 7. Single-Label Root Cause Classifier
        with timer.stage("risk_assessment"):
            root_cause = RootCauseClassifier.classify(
                h_score=overall_h,
                p1_res=p1,
                p2_res=p2,
                p3_res=p3,
                evidence_items=p1.evidence,
                query=query,
                response_text=response_text,
            ).value
        risk_m = timer.measurements.get("risk_assessment")
        risk_ms = risk_m.duration_ms if risk_m else 0.0

        # 8. Response Serialization Timing
        with timer.stage("response_serialization"):
            unc = getattr(report, "uncertainty_analysis", {})
            sig_type = "MEASURED" if (p2_available and payload.logprobs) else ("PROXY" if p2_available else "UNAVAILABLE")
            method_str = "TOKEN_LOGPROBS" if (p2_available and payload.logprobs) else ("UNCERTAINTY_PROXY" if p2_available else "UNAVAILABLE")
            gen_count = len(payload.sample_responses) if payload.sample_responses else 1
            confidence_info = ConfidenceAnalysis(
                whitebox_entropy=float(p2.avg_entropy) if (p2_available and p2.avg_entropy is not None) else None,
                blackbox_variation_score=float(p3.consistency_failure_score) if (p3_available and p3.consistency_failure_score is not None) else None,
                epistemic_uncertainty=float(unc.get("epistemic_uncertainty", 0.0)) if p2_available or p3_available else None,
                aleatoric_uncertainty=float(unc.get("aleatoric_uncertainty", 0.0)) if p2_available else None,
                methodology=method_str,
                signal_type=sig_type,
                uncertainty_measure="Binary Shannon Entropy H(p)" if (p2_available and payload.logprobs) else ("Multi-Sample Variance Proxy" if p2_available else "None"),
                generations_used=gen_count,
                raw_signal_metadata={
                    "logprobs_provided": bool(payload.logprobs),
                    "generations_count": gen_count,
                    "model_name": model_name,
                },
                explanation=p2.reasoning if p2 else "Token-level logprobs not provided by active LLM provider. Excluded from fusion.",
            )

            confidence_val = round(1.0 - abs(overall_h - 0.5), 4)
        ser_m = timer.measurements.get("response_serialization")
        serialization_ms = ser_m.duration_ms if ser_m else 0.0

        # Finalize timer and emit structured JSON logs
        total_req_ms = timer.finish()
        timer.record("total_request", total_req_ms)

        measured_timings_obj = MeasuredTimingBreakdown(
            retrieval_ms=round(p1_dur_ms, 2) if p1_dur_ms > 0 else None,
            bm25_ms=round(float(ret_perf.get("bm25_ms", 0.0)), 2) if ret_perf.get("bm25_ms") else None,
            dense_ms=round(float(ret_perf.get("wikipedia_ms", 0.0)) + float(ret_perf.get("faiss_ms", 0.0)), 2) if (ret_perf.get("wikipedia_ms") or ret_perf.get("faiss_ms")) else None,
            nli_ms=round(float(ret_perf.get("nli_ms", 0.0)) + float(cons_perf.get("nli_ms", 0.0)), 2) if (ret_perf.get("nli_ms") or cons_perf.get("nli_ms")) else None,
            gemini_generation_ms=None,
            p1_latency_ms=round(p1_dur_ms, 2),
            p2_latency_ms=round(p2_dur_ms, 2),
            p3_latency_ms=round(p3_dur_ms, 2),
            fusion_latency_ms=round(fusion_dur_ms, 2),
            total_latency_ms=round(total_req_ms, 2),
        )

        full_perf_dict = timer.get_summary()

        # Record Real Pipeline Trace (No fake derivations)
        tracer.record_stage("pillar1_evidence_grounding", p1_dur_ms, {"factual_error": fe_val, "evidence_count": len(evidence_items)}, confidence=1.0 - fe_val)
        tracer.record_stage("pillar2_confidence_estimation", p2_dur_ms, {"available": p2_available, "confidence_gap": cg_val if p2_available else None}, confidence=1.0 - cg_val if p2_available else None)
        tracer.record_stage("pillar3_consistency_reasoning", p3_dur_ms, {"available": p3_available, "consistency_failure": cf_val if p3_available else None}, confidence=1.0 - cf_val if p3_available else None)
        tracer.record_stage("adaptive_fusion_engine", fusion_dur_ms, {"weights": weights_used, "is_full_analysis": is_full_analysis}, confidence=1.0 - overall_h)
        tracer.finalize(
            final_h_score=overall_h,
            risk_level=risk_level_str,
            root_cause=root_cause,
            metadata={
                "performance_timings": full_perf_dict,
                "measured_timings": measured_timings_obj.model_dump(),
                "pillar_status": p_status_obj.model_dump(),
                "fusion_decomposition": fusion_decomp.model_dump(),
            },
        )

        stage_timings_record = {
            "retrieval_ms": p1_dur_ms,
            "nli_ms": float(ret_perf.get("nli_ms", 0.0)) + float(cons_perf.get("nli_ms", 0.0)),
            "confidence_ms": p2_dur_ms,
            "consistency_ms": p3_dur_ms,
            "fusion_ms": fusion_dur_ms,
            "risk_ms": risk_ms,
            "localization_ms": localization_ms,
            "serialization_ms": serialization_ms,
        }
        _metrics_tracker.record_request(total_req_ms, overall_h, is_success=True, stage_timings=stage_timings_record)

        # Add HTTP Latency Header
        response.headers["X-HalluciSense-Latency-Ms"] = f"{total_req_ms:.2f}"

        res_body = AnalysisResponse(
            trace_id=tracer.trace_id,
            overall_h_score=round(overall_h, 4),
            risk_level=risk_level_str,
            confidence=confidence_val,
            pillar_scores=pillar_scores,
            failure_taxonomy=root_cause if overall_h >= 0.35 else "NONE",
            processing_time_ms=round(total_req_ms, 2),
            version="1.0.0",
            hallucination=overall_h >= 0.54,
            sentence_scores=sentences,
            token_heatmap=heatmap_items,
            evidence=evidence_items,
            confidence_analysis=confidence_info,
            root_cause_classification=root_cause,
            measured_timings=measured_timings_obj,
            pillar_status=p_status_obj,
            fusion_decomposition=fusion_decomp,
        )
        return res_body

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        _metrics_tracker.record_request(0.0, 0.0, is_success=False)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HalluciSense Pipeline Processing Error: {str(e)}",
        )


@router.post(
    "/explain",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed hallucination explainability analysis",
    description="Returns supporting passages, contradiction evidence, token heatmap, sentence scores, reasoning chain, fusion contribution, adaptive weights, and confidence explanation.",
)
async def explain_analysis(payload: ExplainRequest, request: Request, response: Response) -> ExplainResponse:
    """Execute detailed explainability analysis."""
    analysis = await analyze_response(AnalysisRequest(query=payload.query, response=payload.response, model_name=payload.model_name), request, response)

    supporting = [ev.snippet for ev in analysis.evidence if ev.score >= 0.70] or ["No explicit supporting passage retrieved."]
    contradiction = [ev.snippet for ev in analysis.evidence if ev.score < 0.50] or ["No explicit contradiction passage detected."]

    p2_str = f"{analysis.pillar_scores.confidence:.4f}" if analysis.pillar_scores.confidence is not None else "UNAVAILABLE (no logprobs)"
    p3_str = f"{analysis.pillar_scores.consistency:.4f}" if analysis.pillar_scores.consistency is not None else "UNAVAILABLE (single generation)"

    reasoning_chain = [
        f"Step 1: Decomposed response into {len(analysis.sentence_scores)} constituent sentences.",
        f"Step 2: Retrieved {len(analysis.evidence)} grounding passages from reference indices.",
        f"Step 3: Computed Pillar 1 factual error score (FE = {analysis.pillar_scores.retrieval:.4f}).",
        f"Step 4: Evaluated Pillar 2 logit entropy (CG = {p2_str}).",
        f"Step 5: Evaluated Pillar 3 paraphrase consistency (CF = {p3_str}).",
        f"Step 6: Applied Platt recalibrated adaptive fusion resulting in overall H-score = {analysis.overall_h_score:.4f}.",
    ]

    decomp = analysis.fusion_decomposition
    eff_w = decomp.effective_weights if decomp else {"alpha": 0.45, "beta": 0.30, "gamma": 0.25}
    w_contrib = decomp.weighted_contributions if decomp else {}

    fusion_contrib = {
        "pillar1_retrieval_contribution": float(w_contrib.get("p1_contribution") or 0.0),
        "pillar2_confidence_contribution": float(w_contrib.get("p2_contribution") or 0.0),
        "pillar3_consistency_contribution": float(w_contrib.get("p3_contribution") or 0.0),
    }

    adaptive_weights = {
        "alpha_factual_error": float(eff_w.get("alpha", 0.45)),
        "beta_confidence_gap": float(eff_w.get("beta", 0.30)),
        "gamma_consistency_failure": float(eff_w.get("gamma", 0.25)),
    }

    conf_exp = f"Calculated system confidence = {analysis.confidence:.4f}. The output is classified as '{analysis.risk_level}' with root cause '{analysis.root_cause_classification}' ({'Full 3-Pillar Analysis' if (decomp and decomp.is_full_analysis) else 'Partial Analysis'})."

    return ExplainResponse(
        trace_id=analysis.trace_id,
        overall_h_score=analysis.overall_h_score,
        risk_level=analysis.risk_level,
        retrieved_evidence=analysis.evidence,
        supporting_passages=supporting,
        contradiction_evidence=contradiction,
        token_heatmap=analysis.token_heatmap,
        sentence_scores=analysis.sentence_scores,
        reasoning_chain=reasoning_chain,
        fusion_contribution=fusion_contrib,
        adaptive_weights=adaptive_weights,
        confidence_explanation=conf_exp,
        fusion_decomposition=decomp,
        measured_timings=analysis.measured_timings,
    )


@router.get("/debug/latest", summary="Retrieve latest pipeline trace")
async def get_latest_debug_trace() -> Dict[str, Any]:
    """Retrieve the most recent pipeline trace payload."""
    trace = get_latest_trace()
    if not trace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No trace files found.")
    return trace


@router.get("/debug/trace/{trace_id}", summary="Retrieve pipeline trace by ID")
@router.get("/debug/{trace_id}", summary="Retrieve pipeline trace by ID (alias)")
async def get_debug_trace_by_id(trace_id: str) -> Dict[str, Any]:
    """Retrieve pipeline trace payload by trace_id."""
    trace = get_trace_by_id(trace_id)
    if not trace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Trace ID {trace_id} not found.")
    return trace


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Production runtime metrics",
    description="Returns total request counts, average latency, average H-score, success rate, error rate, and process RAM memory.",
)
async def get_production_metrics() -> MetricsResponse:
    """Return runtime metrics computed from real system telemetry."""
    m_dict = _metrics_tracker.get_metrics()
    return MetricsResponse(
        requests=m_dict["requests"],
        average_latency_ms=m_dict["average_latency_ms"],
        average_h_score=m_dict["average_h_score"],
        success_rate=m_dict["success_rate"],
        error_rate=m_dict["error_rate"],
        memory_mb=m_dict["memory_mb"],
    )


from fastapi.responses import PlainTextResponse

@router.get(
    "/metrics/prometheus",
    response_class=PlainTextResponse,
    summary="Prometheus metrics exporter",
    description="Exposes system telemetry metrics formatted for Prometheus scraper compatibility.",
)
async def get_prometheus_metrics() -> PlainTextResponse:
    """Return Prometheus text exposition format metrics."""
    m_dict = _metrics_tracker.get_metrics()
    total_reqs = m_dict["requests"]
    avg_lat_sec = m_dict["average_latency_ms"] / 1000.0
    h_avg = m_dict["average_h_score"]
    succ_pct = m_dict["success_rate"]
    mem_bytes = int(m_dict["memory_mb"] * 1024 * 1024)

    lines = [
        "# HELP hallucisense_requests_total Total number of processed verification requests.",
        "# TYPE hallucisense_requests_total counter",
        f"hallucisense_requests_total {total_reqs}",
        "",
        "# HELP hallucisense_request_latency_seconds Average request processing latency in seconds.",
        "# TYPE hallucisense_request_latency_seconds gauge",
        f"hallucisense_request_latency_seconds {avg_lat_sec:.6f}",
        "",
        "# HELP hallucisense_average_h_score Average hallucination score across requests.",
        "# TYPE hallucisense_average_h_score gauge",
        f"hallucisense_average_h_score {h_avg:.4f}",
        "",
        "# HELP hallucisense_success_rate_percent Request success rate percentage.",
        "# TYPE hallucisense_success_rate_percent gauge",
        f"hallucisense_success_rate_percent {succ_pct:.2f}",
        "",
        "# HELP hallucisense_process_memory_bytes Process resident memory usage in bytes.",
        "# TYPE hallucisense_process_memory_bytes gauge",
        f"hallucisense_process_memory_bytes {mem_bytes}",
    ]
    return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4; charset=utf-8")
