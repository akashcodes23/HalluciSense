// ─────────────────────────────────────────────────────────────────────────────
// HalluciSense v1.0 — React Query Hooks
// All hooks consume real production backend endpoints
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
import { useAnalysisStore } from "@/store/analysis-store";

// ── Analysis Mutation ────────────────────────────────────────────────────────

export function useAnalysis() {
  const addToHistory = useAnalysisStore((s) => s.addToHistory);
  const setCurrentResult = useAnalysisStore((s) => s.setCurrentResult);

  return useMutation({
    mutationFn: (payload: AnalysisRequest) => analyzeResponse(payload),
    onSuccess: (data: AnalysisResponse, variables: AnalysisRequest) => {
      setCurrentResult(data);
      addToHistory({
        id: data.trace_id,
        query: variables.query,
        response: variables.response,
        result: data,
        timestamp: new Date().toISOString(),
      });
    },
  });
}

// ── Explain Mutation ─────────────────────────────────────────────────────────

export function useExplain() {
  const setCurrentExplain = useAnalysisStore((s) => s.setCurrentExplain);

  return useMutation({
    mutationFn: (payload: ExplainRequest) => explainResponse(payload),
    onSuccess: (data: ExplainResponse) => {
      setCurrentExplain(data);
    },
  });
}

// ── Metrics Query (polling every 5s) ─────────────────────────────────────────

export function useMetrics(enabled = true) {
  return useQuery({
    queryKey: ["metrics"],
    queryFn: getMetrics,
    refetchInterval: 5000,
    enabled,
    retry: 2,
  });
}

// ── Health Query (polling every 10s) ─────────────────────────────────────────

export function useHealth(enabled = true) {
  return useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 10000,
    enabled,
    retry: 1,
  });
}

// ── Readiness Query ──────────────────────────────────────────────────────────

export function useReadiness(enabled = true) {
  return useQuery({
    queryKey: ["readiness"],
    queryFn: getReady,
    refetchInterval: 10000,
    enabled,
    retry: 1,
  });
}

// ── Latest Debug Trace ───────────────────────────────────────────────────────

export function useLatestTrace(enabled = true) {
  return useQuery({
    queryKey: ["debug", "latest"],
    queryFn: getLatestDebug,
    enabled,
    retry: 1,
  });
}

// ── Debug Trace by ID ────────────────────────────────────────────────────────

export function useDebugTrace(traceId: string | null) {
  return useQuery({
    queryKey: ["debug", traceId],
    queryFn: () => getDebugTrace(traceId!),
    enabled: !!traceId,
    retry: 1,
  });
}
