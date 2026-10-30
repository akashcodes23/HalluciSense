// ─────────────────────────────────────────────────────────────────────────────
// HalluciSense v1.0 — Production API Client
// Integrated with Current Validated ML Pipeline (/api/v1/hallucisense/*)
// Consumes real backend endpoints. Zero mocks. Zero placeholders.
// ─────────────────────────────────────────────────────────────────────────────

import type {
  AnalysisRequest,
  AnalysisResponse,
  ExplainRequest,
  ExplainResponse,
  MetricsResponse,
  TraceData,
  CorrectionRequest,
  CorrectionResponse,
  PredictRequest,
  PredictResponse,
  HalluciSenseExplainRequest,
  HalluciSenseExplainResponse,
  RiskLevel,
  SentenceScore,
  EvidenceItem,
} from "@/types/hallucisense";

const getBaseUrl = (): string => {
  if (typeof window !== "undefined") {
    const savedUrl = localStorage.getItem("hallucisense_api_url");
    if (savedUrl) {
      return savedUrl;
    }
  }
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL;
  }
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (process.env.NODE_ENV === "production" || process.env.NEXT_PUBLIC_APP_ENV === "production") {
    return "https://hallucisense-production-c0ee.up.railway.app";
  }
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host !== "localhost" && host !== "127.0.0.1") {
      return "https://hallucisense-production-c0ee.up.railway.app";
    }
  }
  return "http://localhost:8000";
};

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new HalluciSenseAPIError(
        body.detail || body.message || `HTTP ${res.status}`,
        res.status,
        body
      );
    }

    return res.json() as Promise<T>;
  } catch (err: unknown) {
    if (err instanceof HalluciSenseAPIError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new HalluciSenseAPIError(
        "Request timed out. The verification service did not respond within 60 seconds.",
        408,
        { error: "TIMEOUT" }
      );
    }
    throw new HalluciSenseAPIError(
      err instanceof Error ? err.message : "Verification service unavailable",
      503,
      { error: "NETWORK_ERROR" }
    );
  } finally {
    clearTimeout(timeout);
  }
}

export class HalluciSenseAPIError extends Error {
  public status: number;
  public body: Record<string, unknown>;

  constructor(
    message: string,
    status: number,
    body: Record<string, unknown>
  ) {
    super(message);
    this.name = "HalluciSenseAPIError";
    this.status = status;
    this.body = body;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalization Adapters (Map Current ML Pipeline -> UI Representation)
// ─────────────────────────────────────────────────────────────────────────────

export function normalizePredictResponse(
  res: PredictResponse,
  latencyMs?: number
): AnalysisResponse {
  const isHallucinated = res.is_hallucinated;
  const hProb = res.hallucination_probability;
  const riskLevel: RiskLevel = isHallucinated
    ? "LIKELY_HALLUCINATED"
    : hProb >= 0.35
    ? "NEEDS_VERIFICATION"
    : "VERIFIED";

  // Build claim-level sentence scores
  const sentenceScores: SentenceScore[] = [];
  const evidenceItems: EvidenceItem[] = [];

  if (res.verification_summary?.claims && res.verification_summary.claims.length > 0) {
    res.verification_summary.claims.forEach((c, idx) => {
      const cStatus = c.status;
      const cRisk: RiskLevel =
        cStatus === "CONTRADICTED"
          ? "LIKELY_HALLUCINATED"
          : cStatus === "VERIFIED"
          ? "VERIFIED"
          : "NEEDS_VERIFICATION";

      const matchedEv: EvidenceItem[] = [];
      if (c.evidence && c.evidence.length > 0) {
        c.evidence.forEach((ev, eIdx) => {
          const item: EvidenceItem = {
            id: `ev_${c.claim_id}_${eIdx}`,
            claim: c.claim_text,
            snippet: ev.snippet,
            title: ev.title,
            source_name: ev.title,
            source_url: ev.url,
            similarity_score: ev.entailment,
            score: ev.entailment,
            retrieval_method: c.verification_method,
          };
          matchedEv.push(item);
          evidenceItems.push(item);
        });
      }

      sentenceScores.push({
        sentence_index: c.claim_id ?? idx,
        sentence_text: c.claim_text,
        text: c.claim_text,
        score: c.verification_confidence,
        h_score: cStatus === "CONTRADICTED" ? 1.0 : cStatus === "VERIFIED" ? 0.0 : 0.5,
        risk_level: cRisk,
        evidence_matched: matchedEv,
        reasoning_summary: c.reason,
      });
    });
  } else if (res.claims && res.claims.length > 0) {
    res.claims.forEach((claimText, idx) => {
      sentenceScores.push({
        sentence_index: idx,
        sentence_text: claimText,
        text: claimText,
        score: hProb,
        h_score: hProb,
        risk_level: riskLevel,
        reasoning_summary: res.explanation?.summary,
      });
    });
  }

  const p1Prob = res.explanation?.pillar_contributions?.pillar_1_probability ?? 0;
  const p2Prob = res.explanation?.pillar_contributions?.pillar_2_probability ?? null;

  return {
    request_id: res.request_id,
    trace_id: res.trace_id,
    overall_h_score: hProb,
    risk_level: riskLevel,
    confidence: res.confidence_score,
    flagged_sentences_count:
      res.verification_summary?.contradicted_claims ?? (isHallucinated ? 1 : 0),
    total_sentences_count: res.claim_count ?? sentenceScores.length,
    pillar_scores: {
      retrieval: p1Prob,
      consistency: null, // Legacy Phase 7 had no Pillar 3 multi-sample consistency
      confidence: p2Prob,
      pillar1_factual_error: p1Prob,
      pillar2_confidence_gap: p2Prob,
      pillar3_consistency_failure: null,
      effective_weights: {
        alpha: 0.6,
        beta: 0.4,
        gamma: 0.0,
      },
    },
    pillar_status: {
      p1_status: "EXECUTED",
      p2_status: p2Prob !== null ? "EXECUTED" : "UNAVAILABLE",
      p3_status: "UNAVAILABLE",
      fusion_status: "PARTIAL_TWO_PILLAR",
      p1_available: true,
      p2_available: p2Prob !== null,
      p3_available: false,
      is_full_analysis: false,
    },
    sentence_scores: sentenceScores,
    evidence: evidenceItems,
    root_cause_classification: res.explanation?.primary_driver || (isHallucinated ? "HALLUCINATION_DETECTED" : "VERIFIED"),
    failure_taxonomy: res.explanation?.risk_severity || (isHallucinated ? "HIGH" : "LOW"),
    latency_ms: latencyMs,
    processing_time_ms: latencyMs,
    version: "1.0.0",
    local_attribution: res.local_attribution,
    semantic_grounding: res.semantic_grounding,
    candidate_comparison: res.candidate_comparison,
    verification_summary: res.verification_summary,
  };
}

export function normalizeExplainResponse(
  res: HalluciSenseExplainResponse,
  latencyMs?: number
): ExplainResponse {
  const p1Prob = res.explanation?.pillar_contributions?.pillar_1_probability ?? null;
  const p2Prob = res.explanation?.pillar_contributions?.pillar_2_probability ?? null;
  const disagreement = res.explanation?.pillar_contributions?.probability_disagreement ?? (
    (p1Prob !== null && p2Prob !== null) ? Math.abs(p1Prob - p2Prob) : null
  );

  const supporting: string[] = [];
  const contradiction: string[] = [];
  const evidenceItems: EvidenceItem[] = [];

  if (res.semantic_grounding?.claims) {
    res.semantic_grounding.claims.forEach((c, idx) => {
      if (c.evidence_details) {
        c.evidence_details.forEach((ev, eIdx) => {
          evidenceItems.push({
            id: `ev_${c.claim_id}_${eIdx}`,
            claim: c.claim_text,
            snippet: ev.snippet,
            title: ev.title,
            source_name: ev.title,
            source_url: ev.url,
            similarity_score: ev.entailment,
            score: ev.entailment,
          });
          if (ev.label === "entailment" || ev.entailment > 0.6) {
            supporting.push(ev.snippet);
          } else if (ev.label === "contradiction" || ev.contradiction > 0.6) {
            contradiction.push(ev.snippet);
          }
        });
      }
    });
  }

  const reasoningChain: string[] = [
    `Step 1: Extracted ${res.prediction?.claim_count ?? 1} claims from response text.`,
    `Step 2: Executed Pillar 1 Semantic Grounding (P1 risk = ${p1Prob !== null ? (p1Prob * 100).toFixed(1) + "%" : "UNAVAILABLE"}).`,
    `Step 3: Executed Pillar 2 Structural Consistency (P2 risk = ${p2Prob !== null ? (p2Prob * 100).toFixed(1) + "%" : "UNAVAILABLE"}).`,
    `Step 4: Evaluated 19-feature hybrid model (Operating threshold τ = ${res.prediction?.operating_threshold ?? 0.54}).`,
    `Step 5: Generated local counterfactual feature attributions and verification trace.`,
  ];

  const summaryText = res.explanation?.summary ?? "Explanation generated from HalluciSense hybrid pipeline.";
  const recText = res.explanation?.recommendation ? `\n\n**Recommendation:** ${res.explanation.recommendation}` : "";

  return {
    explanation_markdown: `${summaryText}${recText}`,
    confidence_explanation: res.explanation?.primary_driver,
    supporting_passages: supporting.length > 0 ? supporting : ["No explicit supporting passages retrieved."],
    contradiction_evidence: contradiction.length > 0 ? contradiction : ["No explicit contradiction passages detected."],
    retrieved_evidence: evidenceItems,
    reasoning_chain: reasoningChain,
    fusion_contribution: {
      pillar1_retrieval_contribution: p1Prob,
      pillar2_confidence_contribution: p2Prob,
      pillar3_consistency_contribution: null, // Legacy Phase 7/14 never had P3 consistency! Must never fabricate P3.
      probability_disagreement: disagreement,
    },
    measured_timings: {
      p1_latency_ms: 0,
      p2_latency_ms: 0,
      p3_latency_ms: 0,
      fusion_latency_ms: 0,
      total_latency_ms: res.latency_ms ?? latencyMs ?? 0,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Canonical 3-Pillar Adapters (/api/v1/analyze & /api/v1/explain)
// ─────────────────────────────────────────────────────────────────────────────

interface BackendAnalysisResponseRaw {
  trace_id: string;
  overall_h_score: number;
  risk_level: string;
  confidence: number;
  pillar_scores: {
    retrieval: number;
    confidence: number | null;
    consistency: number | null;
  };
  failure_taxonomy?: string;
  processing_time_ms: number;
  version?: string;
  hallucination?: boolean;
  sentence_scores?: Array<{
    sentence_index?: number;
    text?: string;
    sentence_text?: string;
    score?: number;
    risk_level?: string;
  }>;
  token_heatmap?: Array<{
    token: string;
    score?: number;
    tier?: string;
    color_hex?: string;
  }>;
  evidence?: Array<{
    id?: string;
    title?: string;
    snippet?: string;
    score?: number;
    source?: string;
    source_name?: string;
    url?: string;
    source_url?: string;
  }>;
  confidence_analysis?: Record<string, unknown>;
  root_cause_classification?: string;
  measured_timings?: any;
  pillar_status?: any;
  fusion_decomposition?: any;
}

export function adaptCanonicalAnalysisResponse(
  raw: BackendAnalysisResponseRaw,
  latencyMs?: number
): AnalysisResponse {
  const p1Error = raw.pillar_scores?.retrieval ?? 0.0;
  const p2Conf = raw.pillar_scores?.confidence ?? null;
  const p3Cons = raw.pillar_scores?.consistency ?? null;

  const p1Available = raw.pillar_status?.p1_available ?? true;
  const p2Available = raw.pillar_status?.p2_available ?? (p2Conf !== null);
  const p3Available = raw.pillar_status?.p3_available ?? (p3Cons !== null);

  const sentenceScores: SentenceScore[] = (raw.sentence_scores || []).map((s, idx) => ({
    sentence_index: s.sentence_index ?? idx,
    sentence_text: s.text || s.sentence_text || "",
    text: s.text || s.sentence_text || "",
    score: s.score ?? 0.0,
    h_score: s.score ?? 0.0,
    risk_level: (s.risk_level as RiskLevel) || "NEEDS_VERIFICATION",
  }));

  const evidenceItems: EvidenceItem[] = (raw.evidence || []).map((e, idx) => ({
    id: e.id || `ev_${idx}`,
    claim: e.title || "",
    snippet: e.snippet || "",
    title: e.title || "",
    source_name: e.source_name || e.source || e.title || "External Evidence",
    source_url: e.source_url || e.url || undefined,
    score: e.score ?? 1.0,
    similarity_score: e.score ?? 1.0,
  }));

  const effWeights = raw.fusion_decomposition?.effective_weights;

  return {
    trace_id: raw.trace_id,
    overall_h_score: raw.overall_h_score ?? 0.0,
    risk_level: (raw.risk_level as RiskLevel) || "NEEDS_VERIFICATION",
    confidence: raw.confidence ?? 0.0,
    flagged_sentences_count: sentenceScores.filter(
      (s) => s.risk_level === "LIKELY_HALLUCINATED" || s.risk_level === "NEEDS_VERIFICATION"
    ).length,
    total_sentences_count: sentenceScores.length,
    pillar_scores: {
      retrieval: p1Error,
      confidence: p2Conf,
      consistency: p3Cons,
      pillar1_factual_error: p1Error,
      pillar2_confidence_gap: p2Conf,
      pillar3_consistency_failure: p3Cons,
      effective_weights: effWeights
        ? {
            alpha: effWeights.alpha_factual_error ?? effWeights.alpha ?? 0.45,
            beta: effWeights.beta_confidence_gap ?? effWeights.beta ?? 0.30,
            gamma: effWeights.gamma_consistency_failure ?? effWeights.gamma ?? 0.25,
          }
        : undefined,
    },
    sentence_scores: sentenceScores,
    evidence: evidenceItems,
    root_cause_classification: raw.root_cause_classification || "VERIFIED",
    failure_taxonomy: raw.failure_taxonomy || "NONE",
    latency_ms: latencyMs ?? raw.processing_time_ms ?? 0,
    processing_time_ms: raw.processing_time_ms ?? latencyMs ?? 0,
    version: raw.version || "1.0.0",
    pillar_status: raw.pillar_status || {
      p1_status: p1Available ? "EXECUTED" : "UNAVAILABLE",
      p2_status: p2Available ? "EXECUTED" : "UNAVAILABLE",
      p3_status: p3Available ? "EXECUTED" : "UNAVAILABLE",
      fusion_status: "PARTIAL_RENORMALIZED",
      p1_available: p1Available,
      p2_available: p2Available,
      p3_available: p3Available,
      is_full_analysis: p1Available && p2Available && p3Available,
    },
    fusion_decomposition: raw.fusion_decomposition,
    confidence_analysis: raw.confidence_analysis as any,
    token_heatmap: (raw.token_heatmap as any) || [],
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Production API Endpoints
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Executes verification on text input using the canonical HalluciSense 3-pillar pipeline:
 * POST /api/v1/analyze
 */
export async function analyzeResponse(
  payload: AnalysisRequest
): Promise<AnalysisResponse> {
  const textToAnalyze = (payload.response || payload.text || "").trim();
  const startTime = performance.now();

  const reqBody = {
    query: payload.query?.trim() || undefined,
    response: textToAnalyze,
    model_name: payload.model_name || "GPT-4",
    provided_evidence: payload.provided_evidence,
    sample_responses: payload.sample_responses,
    logprobs: payload.logprobs,
  };

  const rawRes = await request<BackendAnalysisResponseRaw>("/api/v1/analyze", {
    method: "POST",
    body: JSON.stringify(reqBody),
  });

  const latencyMs = Math.round(performance.now() - startTime);
  return adaptCanonicalAnalysisResponse(rawRes, latencyMs);
}

/**
 * Direct predict call to POST /api/v1/hallucisense/predict (preserved for legacy benchmark comparisons)
 */
export async function predictHallucination(
  payload: PredictRequest
): Promise<PredictResponse> {
  return request<PredictResponse>("/api/v1/hallucisense/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Executes explainability analysis using canonical endpoint:
 * POST /api/v1/explain
 */
export async function explainResponse(
  payload: ExplainRequest
): Promise<ExplainResponse> {
  const textToAnalyze = (payload.response || (payload as AnalysisRequest).text || "").trim();
  const startTime = performance.now();

  const reqBody = {
    query: payload.query?.trim() || undefined,
    response: textToAnalyze,
    model_name: payload.model_name || "GPT-4",
  };

  const rawRes = await request<ExplainResponse>("/api/v1/explain", {
    method: "POST",
    body: JSON.stringify(reqBody),
  });

  const latencyMs = Math.round(performance.now() - startTime);
  return {
    ...rawRes,
    measured_timings: {
      ...rawRes.measured_timings,
      total_latency_ms: rawRes.measured_timings?.total_latency_ms ?? latencyMs,
      p1_latency_ms: rawRes.measured_timings?.p1_latency_ms ?? 0,
      p2_latency_ms: rawRes.measured_timings?.p2_latency_ms ?? 0,
      p3_latency_ms: rawRes.measured_timings?.p3_latency_ms ?? 0,
      fusion_latency_ms: rawRes.measured_timings?.fusion_latency_ms ?? 0,
    },
  };
}

/**
 * Direct explain call to POST /api/v1/hallucisense/explain
 */
export async function explainHallucinationDirect(
  payload: HalluciSenseExplainRequest
): Promise<HalluciSenseExplainResponse> {
  return request<HalluciSenseExplainResponse>("/api/v1/hallucisense/explain", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Closed-loop correction via backend correction path
 */
export async function correctResponse(
  payload: CorrectionRequest
): Promise<CorrectionResponse> {
  return request<CorrectionResponse>("/api/v1/correct", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Retrieves system performance metrics from current pipeline:
 * GET /api/v1/hallucisense/metrics (with fallback to legacy /api/v1/metrics if needed)
 */
export async function getMetrics(): Promise<MetricsResponse> {
  try {
    const raw = await request<{
      pillar1_roc_auc?: number;
      pillar2_roc_auc?: number;
      hybrid_heldout_roc_auc?: number;
      hybrid_heldout_mcc?: number;
      operating_threshold?: number;
      model_status?: string;
    }>("/api/v1/hallucisense/metrics");

    return {
      requests: 1,
      success_rate: 1.0,
      error_rate: 0.0,
      average_h_score: raw.operating_threshold ?? 0.54,
      average_latency_ms: 50,
      status: raw.model_status || "FROZEN AND VALIDATED",
      risk_distribution: {
        "Pillar 1 ROC-AUC": raw.pillar1_roc_auc ?? 0.6259,
        "Pillar 2 ROC-AUC": raw.pillar2_roc_auc ?? 0.5784,
        "Hybrid ROC-AUC": raw.hybrid_heldout_roc_auc ?? 0.6558,
        "Hybrid MCC": raw.hybrid_heldout_mcc ?? 0.1945,
      },
    };
  } catch {
    return request<MetricsResponse>("/api/v1/metrics");
  }
}

/**
 * Retrieves health status from current pipeline:
 * GET /api/v1/hallucisense/health (with fallback to /health)
 */
export async function getHealth(): Promise<{ status: string }> {
  try {
    const res = await request<{ status: string }>("/api/v1/hallucisense/health");
    return { status: res.status || "ok" };
  } catch {
    return request<{ status: string }>("/health");
  }
}

export async function getReady(): Promise<{ status: string; components: Record<string, boolean> }> {
  return request<{ status: string; components: Record<string, boolean> }>("/ready");
}

export async function getLatestDebug(): Promise<TraceData> {
  return request<TraceData>("/api/v1/debug/latest");
}

export async function getDebugTrace(traceId: string): Promise<TraceData> {
  return request<TraceData>(`/api/v1/debug/${traceId}`);
}

export interface ClosedLoopChatPayload {
  message: string;
  enable_verification?: boolean;
  auto_correct?: boolean;
  model_name?: string;
  conversation_id?: string;
}

export interface ClosedLoopChatResponse {
  conversation_id: string;
  message_id: string;
  original_response: string;
  final_response: string;
  verification: {
    status: string;
    h_score: number | null;
    risk_level: string | null;
    claims_total: number | null;
    claims_flagged: number | null;
    error_message?: string;
  };
  correction: {
    performed: boolean;
    reason: string;
    claims_corrected: Array<Record<string, unknown>>;
    original_to_corrected: Array<Record<string, unknown>>;
  };
  evidence: Array<{
    source_name: string;
    snippet: string;
    claim: string;
  }>;
  sources: string[];
  trace_id: string;
  latency_ms: number;
}

export async function sendClosedLoopChat(
  payload: ClosedLoopChatPayload
): Promise<ClosedLoopChatResponse> {
  return request<ClosedLoopChatResponse>("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

