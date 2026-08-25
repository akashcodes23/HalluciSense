// ─────────────────────────────────────────────────────────────────────────────
// HalluciSense v1.0 — React Query Hooks for Verification & Analysis
// ─────────────────────────────────────────────────────────────────────────────

"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import {
  analyzeResponse,
  explainResponse,
  getMetrics,
  getHealth,
  getReady,
  getLatestDebug,
  getDebugTrace,
  HalluciSenseAPIError,
} from "@/services/hallucisense-api";
import type {
  AnalysisRequest,
  AnalysisResponse,
  ExplainRequest,
  ExplainResponse,
} from "@/types/hallucisense";
import { useAnalysisStore, type ErrorEventRiskLevel } from "@/store/analysis-store";

// ─── Risk level normalization ─────────────────────────────────────────────────
/**
 * Maps backend RiskLevel strings to ErrorEventRiskLevel.
 * Returns null for VERIFIED — those must NOT enter the feed.
 */
function normalizeRiskLevel(rl: string | undefined): ErrorEventRiskLevel | null {
  if (!rl) return "NEEDS_VERIFICATION";
  const map: Record<string, ErrorEventRiskLevel | null> = {
    LIKELY_HALLUCINATED: "LIKELY_HALLUCINATED",
    MODERATE_RISK: "MODERATE_RISK",
    NEEDS_VERIFICATION: "NEEDS_VERIFICATION",
    VERIFIED: null,          // intentionally excluded
    CORRECTED: "CORRECTED",
    FAILED: "FAILED",
    REVIEW: "REVIEW",
  };
  // Unknown values default to NEEDS_VERIFICATION
  return rl in map ? map[rl] : "NEEDS_VERIFICATION";
}

// ─── Normalize error message from HalluciSenseAPIError ───────────────────────
function normalizeErrorMessage(err: unknown): string {
  if (err instanceof HalluciSenseAPIError) {
    // Use the API-provided message; include HTTP status for context.
    // Do not expose raw stack traces.
    return `[HTTP ${err.status}] ${err.message}`;
  }
  if (err instanceof Error) {
    return err.message;
  }
  return "An unexpected error occurred.";
}

// ─── Hooks ────────────────────────────────────────────────────────────────────
export function useAnalysis() {
  const addToHistory = useAnalysisStore((s) => s.addToHistory);
  const setCurrentResult = useAnalysisStore((s) => s.setCurrentResult);
  const setIsAnalyzing = useAnalysisStore((s) => s.setIsAnalyzing);
  const addErrorEvent = useAnalysisStore((s) => s.addErrorEvent);

  return useMutation({
    mutationFn: async (payload: AnalysisRequest) => {
      setIsAnalyzing(true);
      const textToAnalyze = payload.text || payload.response || "";
      const reqPayload = {
        query: payload.query?.trim() || undefined,
        response: textToAnalyze.trim(),
        model_name: payload.model_name || "GPT-4",
      };
      return analyzeResponse(reqPayload as AnalysisRequest);
    },
    onSuccess: (data: AnalysisResponse, variables: AnalysisRequest) => {
      setIsAnalyzing(false);
      setCurrentResult(data);
      const historyEntry = {
        id: data.trace_id || `trace_${Date.now()}`,
        query: variables.query?.trim() || "(no query provided)",
        response: variables.text || variables.response || "",
        result: data,
        timestamp: new Date().toISOString(),
      };
      addToHistory(historyEntry);

      // Only push anomalies to the error feed — VERIFIED results are skipped.
      const riskLevel = normalizeRiskLevel(data.risk_level);
      if (riskLevel === null) return; // VERIFIED — do not record

      addErrorEvent({
        id: `verify_${data.trace_id || Date.now()}`,
        timestamp: new Date().toISOString(),
        source: "VERIFY",
        risk_level: riskLevel,
        query: historyEntry.query,
        response: historyEntry.response,
        h_score: data.overall_h_score,
        root_cause: data.root_cause_classification,
        failure_taxonomy: data.failure_taxonomy,
        pillar_scores: data.pillar_scores,
        trace_id: data.trace_id,
        latency_ms: data.processing_time_ms ?? data.latency_ms,
      });
    },
    onError: (err: unknown, variables: AnalysisRequest) => {
      setIsAnalyzing(false);
      addErrorEvent({
        id: `verify_fail_${Date.now()}`,
        timestamp: new Date().toISOString(),
        source: "VERIFY",
        risk_level: "FAILED",
        query: variables.query?.trim() || "(no query provided)",
        response: variables.text || variables.response || "",
        error_message: normalizeErrorMessage(err),
      });
    },
  });
}

export function useExplain() {
  const setCurrentExplain = useAnalysisStore((s) => s.setCurrentExplain);

  return useMutation({
    mutationFn: async (payload: ExplainRequest) => {
      const textToAnalyze = payload.response || (payload as AnalysisRequest).text || "";
      const reqPayload = {
        query: payload.query?.trim() || undefined,
        response: textToAnalyze.trim(),
        model_name: payload.model_name || "GPT-4",
      };
      return explainResponse(reqPayload as ExplainRequest);
    },
    onSuccess: (data: ExplainResponse) => {
      setCurrentExplain(data);
    },
  });
}

export function useMetrics() {
  return useQuery({
    queryKey: ["metrics"],
    queryFn: getMetrics,
    refetchInterval: 5000,
    staleTime: 2000,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 10000,
  });
}

export function useReady() {
  return useQuery({
    queryKey: ["ready"],
    queryFn: getReady,
    refetchInterval: 10000,
  });
}

export function useLatestDebug() {
  return useQuery({
    queryKey: ["debug", "latest"],
    queryFn: getLatestDebug,
    staleTime: 5000,
    retry: false,
  });
}

export const useLatestTrace = useLatestDebug;

export function useDebugTrace(traceId: string | null) {
  return useQuery({
    queryKey: ["debug", traceId],
    queryFn: () => getDebugTrace(traceId!),
    enabled: !!traceId,
    retry: false,
  });
}
