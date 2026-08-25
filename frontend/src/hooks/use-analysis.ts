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
} from "@/services/hallucisense-api";
import type {
  AnalysisRequest,
  AnalysisResponse,
  ExplainRequest,
  ExplainResponse,
} from "@/types/hallucisense";
import { useAnalysisStore, type ErrorEventRiskLevel } from "@/store/analysis-store";

// Map backend RiskLevel to ErrorEventRiskLevel (they overlap but backend uses
// a slightly different vocabulary on the chat path).
function normalizeRiskLevel(rl: string | undefined): ErrorEventRiskLevel {
  if (!rl) return "NEEDS_VERIFICATION";
  const map: Record<string, ErrorEventRiskLevel> = {
    LIKELY_HALLUCINATED: "LIKELY_HALLUCINATED",
    MODERATE_RISK: "MODERATE_RISK",
    NEEDS_VERIFICATION: "NEEDS_VERIFICATION",
    VERIFIED: "VERIFIED",
    CORRECTED: "CORRECTED",
    FAILED: "FAILED",
    REVIEW: "REVIEW",
  };
  return map[rl] ?? "NEEDS_VERIFICATION";
}

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

      // Push to global error feed
      addErrorEvent({
        id: `verify_${data.trace_id || Date.now()}`,
        timestamp: new Date().toISOString(),
        source: "VERIFY",
        risk_level: normalizeRiskLevel(data.risk_level),
        query: historyEntry.query,
        response: historyEntry.response,
        h_score: data.overall_h_score,
        root_cause: data.root_cause_classification,
        pillar_scores: data.pillar_scores,
        trace_id: data.trace_id,
      });
    },
    onError: (_err, variables: AnalysisRequest) => {
      setIsAnalyzing(false);
      // Record system-level failure in the feed
      addErrorEvent({
        id: `verify_fail_${Date.now()}`,
        timestamp: new Date().toISOString(),
        source: "VERIFY",
        risk_level: "FAILED",
        query: variables.query?.trim() || "(no query provided)",
        response: variables.text || variables.response || "",
        error_message: "Verification service returned an error.",
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
