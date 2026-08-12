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
import { useAnalysisStore } from "@/store/analysis-store";

export function useAnalysis() {
  const addToHistory = useAnalysisStore((s) => s.addToHistory);
  const setCurrentResult = useAnalysisStore((s) => s.setCurrentResult);
  const setIsAnalyzing = useAnalysisStore((s) => s.setIsAnalyzing);

  return useMutation({
    mutationFn: async (payload: AnalysisRequest) => {
      setIsAnalyzing(true);
      const textToAnalyze = payload.text || payload.response || "";
      const reqPayload = {
        query: payload.query?.trim() || "General Factual Verification",
        response: textToAnalyze.trim(),
        model_name: payload.model_name || "GPT-4",
      };
      return analyzeResponse(reqPayload as AnalysisRequest);
    },
    onSuccess: (data: AnalysisResponse, variables: AnalysisRequest) => {
      setIsAnalyzing(false);
      setCurrentResult(data);
      addToHistory({
        id: data.trace_id || `trace_${Date.now()}`,
        query: variables.query?.trim() || "General Factual Verification",
        response: variables.text || variables.response || "",
        result: data,
        timestamp: new Date().toISOString(),
      });
    },
    onError: () => {
      setIsAnalyzing(false);
    },
  });
}

export function useExplain() {
  const setCurrentExplain = useAnalysisStore((s) => s.setCurrentExplain);

  return useMutation({
    mutationFn: async (payload: ExplainRequest) => {
      const textToAnalyze = payload.response || (payload as AnalysisRequest).text || "";
      const reqPayload = {
        query: payload.query?.trim() || "General Factual Verification",
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
  });
}

export const useLatestTrace = useLatestDebug;

export function useDebugTrace(traceId: string | null) {
  return useQuery({
    queryKey: ["debug", traceId],
    queryFn: () => getDebugTrace(traceId!),
    enabled: !!traceId,
  });
}
