"use client";

import React, { useState, useEffect, useMemo, Suspense } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  GitBranch,
  Clock,
  Search,
  RefreshCw,
  ChevronDown,
  Layers,
  AlertTriangle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { NoTraces } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { useLatestTrace, useDebugTrace } from "@/hooks/use-analysis";
import { useAnalysisStore } from "@/store/analysis-store";
import { formatLatency, formatTimestamp, formatScore, getRiskColor } from "@/lib/format";
import { cn } from "@/lib/utils";
import { useRouter, useSearchParams } from "next/navigation";
import type { TraceStage, TraceData, AnalysisHistoryEntry } from "@/types/hallucisense";

export default function TracesPage() {
  return (
    <Suspense fallback={<div className="p-6 text-xs text-[var(--text-dim)]">Loading trace explorer…</div>}>
      <ErrorBoundary
        fallbackTitle="Trace Explorer Unavailable"
        fallbackMessage="An unexpected error occurred while rendering the trace explorer. The verification result remains available."
      >
        <TracesContent />
      </ErrorBoundary>
    </Suspense>
  );
}

function TracesContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const [traceIdInput, setTraceIdInput] = useState(searchParams?.get("id") || "");
  const [searchId, setSearchId] = useState<string | null>(searchParams?.get("id") || null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  const history = useAnalysisStore((s) => s.history);
  const selectedTraceId = useAnalysisStore((s) => s.selectedTraceId);
  const setSelectedTraceId = useAnalysisStore((s) => s.setSelectedTraceId);

  // Sync from store
  useEffect(() => {
    if (selectedTraceId) {
      setTraceIdInput(selectedTraceId);
      setSearchId(selectedTraceId);
      setSelectedTraceId(null);
    }
  }, [selectedTraceId, setSelectedTraceId]);

  const { data: latestTrace, isLoading: latestLoading, refetch, isError: latestError } = useLatestTrace();
  const { data: searchedTrace, isLoading: searchLoading, isError: searchError } = useDebugTrace(searchId);

  // Determine and normalize active trace
  const activeTrace = useMemo<TraceData | null>(() => {
    if (searchId) {
      if (searchedTrace) return normalizeTraceData(searchedTrace);
      // Fallback to local history
      const entry = history.find((h) => h.id === searchId || h.result.trace_id === searchId);
      if (entry) return normalizeTraceData(transformHistoryEntry(entry));
    }
    if (latestTrace) return normalizeTraceData(latestTrace);
    if (history.length > 0) return normalizeTraceData(transformHistoryEntry(history[0]));
    return null;
  }, [searchId, searchedTrace, latestTrace, history]);

  const isLoading = searchId ? searchLoading && !activeTrace : latestLoading && !activeTrace;
  const isApiFailed = searchId ? searchError && !activeTrace : latestError && !activeTrace;

  const handleSearch = () => {
    if (traceIdInput.trim()) {
      setSearchId(traceIdInput.trim());
    }
  };

  const toggleNode = (key: string) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // Compute max duration for waterfall bar widths safely
  const maxDuration = useMemo(() => {
    if (!activeTrace?.stages || !Array.isArray(activeTrace.stages) || activeTrace.stages.length === 0) return 1;
    const durations = activeTrace.stages.map((s) => (typeof s.duration_ms === "number" ? s.duration_ms : 0));
    return Math.max(...durations, 1);
  }, [activeTrace]);

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Trace List Sidebar ──────────────────────────────────────── */}
      <div className="hidden lg:flex flex-col w-[280px] border-r border-[var(--border)] bg-[var(--bg-surface)] shrink-0">
        <div className="p-4 border-b border-[var(--border)]">
          <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">Recent Traces</h3>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--text-dim)]" />
            <input
              type="text"
              value={traceIdInput}
              onChange={(e) => setTraceIdInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search trace ID…"
              className={cn(
                "w-full pl-8 pr-3 py-1.5 rounded-[var(--radius-sm)]",
                "bg-[var(--surface)] border border-[var(--border)]",
                "text-xs text-[var(--text-primary)] placeholder:text-[var(--text-dim)]",
                "focus:outline-none focus:border-[var(--primary)]"
              )}
            />
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
          {history.length > 0 ? (
            history.slice(0, 30).map((entry) => {
              const traceId = entry.result.trace_id || entry.id;
              const isActive = searchId === traceId || searchId === entry.id;
              return (
                <button
                  key={entry.id}
                  onClick={() => { setSearchId(traceId); setTraceIdInput(traceId); }}
                  className={cn(
                    "w-full text-left p-2.5 rounded-[var(--radius)] transition-colors cursor-pointer",
                    isActive ? "bg-[var(--primary-soft)] border border-[var(--ai-border)]" : "hover:bg-[var(--surface-hover)] border border-transparent"
                  )}
                >
                  <div className="flex items-center gap-2">
                    <StatusBadge status={entry.result.risk_level} size="sm" />
                    <span className="text-[11px] font-mono text-[var(--text-dim)] truncate">{formatScore(entry.result.overall_h_score)}%</span>
                  </div>
                  <p className="text-[11px] text-[var(--text-muted)] truncate mt-1">
                    {truncate(entry.query || entry.response, 40)}
                  </p>
                  <p className="text-[10px] text-[var(--text-dim)] mt-0.5">{formatTimestamp(entry.timestamp)}</p>
                </button>
              );
            })
          ) : (
            <NoTraces variant="compact" />
          )}
        </div>
      </div>

      {/* ── Trace Detail ────────────────────────────────────────────── */}
      <div className="flex-1 min-w-0 overflow-y-auto p-5 md:p-6 pb-20 md:pb-6 space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-heading-lg text-[var(--text-primary)]">Traces</h1>
            <p className="text-label-md text-[var(--text-muted)] mt-1">
              Inspect stage latency, memory footprint, and decision signals
            </p>
          </div>
          <Button variant="ghost" size="icon-sm" onClick={() => refetch()} aria-label="Refresh">
            <RefreshCw className={cn("w-4 h-4", latestLoading && "animate-spin")} />
          </Button>
        </div>

        {/* Mobile Search */}
        <div className="lg:hidden">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-dim)]" />
            <input
              type="text"
              value={traceIdInput}
              onChange={(e) => setTraceIdInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search trace ID…"
              className={cn(
                "w-full pl-9 pr-3 py-2 rounded-[var(--radius)]",
                "bg-[var(--bg-surface)] border border-[var(--border)]",
                "text-sm text-[var(--text-primary)] placeholder:text-[var(--text-dim)]",
                "focus:outline-none focus:border-[var(--primary)]"
              )}
            />
          </div>
        </div>

        {isLoading ? (
          <div className="space-y-3">
            <p className="text-xs text-[var(--text-muted)] italic">Loading verification trace…</p>
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : isApiFailed ? (
          <div className="p-6 rounded-[var(--radius-lg)] bg-[var(--surface)] border border-[var(--border)] space-y-3 text-center">
            <AlertTriangle className="w-8 h-8 text-amber-400 mx-auto" />
            <h3 className="text-sm font-semibold text-[var(--text-primary)]">Trace unavailable</h3>
            <p className="text-xs text-[var(--text-muted)] max-w-md mx-auto">
              Unable to retrieve verification trace payload from the service. The verification decision remains unaffected.
            </p>
            <Button variant="outline" size="sm" onClick={() => router.push("/verify")}>
              Return to Verify
            </Button>
          </div>
        ) : activeTrace ? (
          <ErrorBoundary fallbackTitle="Trace Visualization Error" fallbackMessage="Could not render execution waterfall. Trace summary is preserved.">
            {/* Trace Summary */}
            <Card>
              <CardContent className="p-4">
                <div className="flex flex-wrap items-center gap-4">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Trace ID</p>
                    <p className="text-sm font-mono text-[var(--text-primary)]">{activeTrace.trace_id}</p>
                  </div>
                  {activeTrace.summary && (
                    <>
                      <div>
                        <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">H-Score</p>
                        <p className="text-lg font-bold font-mono" style={{ color: getRiskColor(activeTrace.summary.risk_level) }}>
                          {formatScore(activeTrace.summary.final_h_score)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Total Latency</p>
                        <p className="text-sm font-mono text-[var(--text-primary)]">
                          {typeof activeTrace.summary.total_duration_ms === "number" ? formatLatency(activeTrace.summary.total_duration_ms) : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Verdict</p>
                        <StatusBadge status={activeTrace.summary.risk_level} size="sm" />
                      </div>
                      {activeTrace.summary.root_cause_classification && activeTrace.summary.root_cause_classification !== "VERIFIED" && activeTrace.summary.root_cause_classification !== "NONE" && (
                        <div>
                          <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Root Cause</p>
                          <Badge variant="hallucination">{activeTrace.summary.root_cause_classification.replace(/_/g, " ")}</Badge>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Waterfall Visualization */}
            {activeTrace.stages && activeTrace.stages.length > 0 ? (
              <div>
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-3">
                  Pipeline Execution Waterfall
                </h3>
                <div className="space-y-1">
                  {activeTrace.stages.map((stage, i) => {
                    const isExpanded = expandedNodes.has(stage.name);
                    const durationVal = typeof stage.duration_ms === "number" ? stage.duration_ms : 0;
                    const barWidth = durationVal > 0 ? Math.max((durationVal / maxDuration) * 100, 4) : 0;
                    const statusColor = getStageColor(stage.status);

                    return (
                      <div key={stage.name || i}>
                        <button
                          onClick={() => toggleNode(stage.name)}
                          className="w-full text-left hover:bg-[var(--surface-hover)] rounded-[var(--radius)] p-2 transition-colors cursor-pointer"
                        >
                          <div className="flex items-center gap-3">
                            {/* Status Dot */}
                            <span
                              className="w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: statusColor }}
                            />
                            {/* Stage Name */}
                            <span className="text-[13px] text-[var(--text-primary)] flex-1 min-w-0 truncate">
                              {stage.name}
                            </span>
                            {/* Duration */}
                            <span className="text-[11px] font-mono text-[var(--text-muted)] shrink-0">
                              {stage.duration_ms != null ? formatLatency(stage.duration_ms) : "—"}
                            </span>
                            {/* Status Badge */}
                            <Badge
                              variant={stage.status === "completed" || stage.status === "success" ? "verified" : stage.status === "failed" ? "hallucination" : "outline"}
                              size="sm"
                            >
                              {stage.status}
                            </Badge>
                            <ChevronDown className={cn("w-3.5 h-3.5 text-[var(--text-dim)] transition-transform", isExpanded && "rotate-180")} />
                          </div>
                          {/* Waterfall Bar */}
                          {barWidth > 0 && (
                            <div className="mt-1.5 ml-5 h-1.5 bg-[var(--surface)] rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${barWidth}%` }}
                                transition={{ duration: 0.4, delay: i * 0.05 }}
                                className="h-full rounded-full"
                                style={{ backgroundColor: statusColor }}
                              />
                            </div>
                          )}
                        </button>

                        {/* Expanded Details */}
                        <AnimatePresence>
                          {isExpanded && stage.details && Object.keys(stage.details).length > 0 && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.15 }}
                              className="overflow-hidden"
                            >
                              <div className="ml-7 mb-2 p-3 rounded-[var(--radius)] bg-[var(--surface)] border border-[var(--border)]">
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[12px]">
                                  {Object.entries(stage.details).map(([key, val]) => (
                                    <div key={key} className="truncate">
                                      <span className="text-[var(--text-dim)]">{key.replace(/_/g, " ")}:</span>{" "}
                                      <span className="text-[var(--text-secondary)] font-mono">
                                        {typeof val === "number" ? val.toFixed(4) : typeof val === "object" && val !== null ? JSON.stringify(val) : String(val ?? "—")}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-[var(--radius)] bg-[var(--surface)] border border-[var(--border)] text-xs text-[var(--text-dim)]">
                Trace data is incomplete. The verification result remains available.
              </div>
            )}

            {/* Measured Timings */}
            {activeTrace.measured_timings && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Clock className="w-4 h-4 text-[var(--text-muted)]" />
                    Measured Stage Latencies
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {Object.entries(activeTrace.measured_timings).map(([key, val]) => {
                      if (val == null) return null;
                      return (
                        <div key={key} className="rounded-[var(--radius)] bg-[var(--surface)] p-2 text-center">
                          <p className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider mb-0.5">
                            {key.replace(/_ms$/, "").replace(/_/g, " ")}
                          </p>
                          <p className="text-sm font-mono font-medium text-[var(--text-secondary)]">
                            {typeof val === "number" ? formatLatency(val) : String(val)}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Fusion Decomposition (if available) */}
            {activeTrace.fusion_decomposition && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Layers className="w-4 h-4 text-[var(--text-muted)]" />
                    Mathematical Signal Fusion
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="ai" size="sm">{activeTrace.fusion_decomposition.fusion_mode}</Badge>
                    <span className="text-xs font-mono text-[var(--text-muted)]">
                      {activeTrace.fusion_decomposition.equation}
                    </span>
                  </div>
                  {activeTrace.fusion_decomposition.explanation && (
                    <p className="text-xs text-[var(--text-dim)] leading-relaxed">
                      {activeTrace.fusion_decomposition.explanation}
                    </p>
                  )}
                </CardContent>
              </Card>
            )}
          </ErrorBoundary>
        ) : (
          <NoTraces onNavigate={() => router.push("/verify")} />
        )}
      </div>
    </div>
  );
}

/**
 * Format raw backend stage names into canonical human-readable labels.
 */
function formatCanonicalStageName(rawName: string): string {
  const map: Record<string, string> = {
    pillar1_evidence_grounding: "Pillar 1 — Evidence Support",
    pillar1_evidence_support: "Pillar 1 — Evidence Support",
    pillar2_confidence_estimation: "Pillar 2 — Model Uncertainty",
    pillar2_model_uncertainty: "Pillar 2 — Model Uncertainty",
    pillar3_consistency_reasoning: "Pillar 3 — Generation Consistency",
    pillar3_generation_consistency: "Pillar 3 — Generation Consistency",
    adaptive_fusion_engine: "Adaptive Hybrid Fusion",
    adaptive_fusion: "Adaptive Hybrid Fusion",
    input_validation: "Input Validation",
    claim_decomposition: "Claim Decomposition",
    evidence_retrieval: "Evidence Retrieval",
    token_localization: "Token Localization",
    root_cause_classification: "Root Cause Classification",
  };

  if (rawName in map) return map[rawName];

  return rawName
    .replace(/^pillar_?1[_-]?/i, "Pillar 1 — ")
    .replace(/^pillar_?2[_-]?/i, "Pillar 2 — ")
    .replace(/^pillar_?3[_-]?/i, "Pillar 3 — ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Normalizes any trace data structure (from API dict or history array) into a safe TraceData shape.
 */
function normalizeTraceData(raw: unknown): TraceData | null {
  if (!raw || typeof raw !== "object") return null;

  const obj = raw as Record<string, unknown>;
  const traceId = typeof obj.trace_id === "string" ? obj.trace_id : `TRACE_${Date.now()}`;
  const timestamp = typeof obj.timestamp === "string" ? obj.timestamp : new Date().toISOString();

  let stagesList: TraceStage[] = [];

  if (Array.isArray(obj.stages)) {
    stagesList = obj.stages.map((st) => {
      if (!st || typeof st !== "object") {
        console.warn("[normalizeTraceData] Skipping malformed stage element:", st);
        return { name: "Stage", status: "completed" as const, duration_ms: null };
      }
      const stObj = st as Record<string, unknown>;
      return {
        name: formatCanonicalStageName(String(stObj.name || "Pipeline Stage")),
        status: (stObj.status as TraceStage["status"]) || "completed",
        duration_ms: typeof stObj.duration_ms === "number" ? stObj.duration_ms : null,
        memory_mb: typeof stObj.memory_mb === "number" ? stObj.memory_mb : undefined,
        details: typeof stObj.details === "object" && stObj.details !== null ? (stObj.details as Record<string, unknown>) : undefined,
      };
    });
  } else if (obj.stages !== null && typeof obj.stages === "object") {
    // Backend dictionary mapping stage_name -> stage data
    stagesList = Object.entries(obj.stages as Record<string, Record<string, unknown>>).map(([stageKey, stageVal]) => {
      if (!stageVal || typeof stageVal !== "object") {
        console.warn("[normalizeTraceData] Malformed stage value for key:", stageKey, stageVal);
        return {
          name: formatCanonicalStageName(stageKey),
          status: "unavailable" as const,
          duration_ms: null,
        };
      }
      const dur = typeof stageVal?.duration_ms === "number" ? stageVal.duration_ms : null;
      const mem = typeof stageVal?.memory_mb === "number" ? stageVal.memory_mb : undefined;
      const details = typeof stageVal?.details === "object" && stageVal.details !== null ? (stageVal.details as Record<string, unknown>) : undefined;
      const isUnavailable = details?.available === false || details?.reason != null;

      return {
        name: formatCanonicalStageName(stageKey),
        status: isUnavailable ? "unavailable" : dur !== null && dur >= 0 ? "completed" : "completed",
        duration_ms: dur,
        memory_mb: mem,
        details,
      };
    });
  } else if (obj.stages !== undefined && obj.stages !== null) {
    console.warn("[normalizeTraceData] Unrecognized stages field type:", typeof obj.stages, obj.stages);
  }

  const rawSummary = typeof obj.summary === "object" && obj.summary !== null ? (obj.summary as Record<string, unknown>) : {};
  const totalDuration = typeof rawSummary.total_duration_ms === "number"
    ? rawSummary.total_duration_ms
    : stagesList.reduce((acc, s) => acc + (s.duration_ms || 0), 0);

  const summary = {
    total_duration_ms: totalDuration,
    total_memory_mb: typeof rawSummary.total_memory_mb === "number" ? rawSummary.total_memory_mb : 0,
    final_h_score: typeof rawSummary.final_h_score === "number" ? rawSummary.final_h_score : 0,
    risk_level: typeof rawSummary.risk_level === "string" ? rawSummary.risk_level : "NEEDS_VERIFICATION",
    root_cause_classification: typeof rawSummary.root_cause_classification === "string" ? rawSummary.root_cause_classification : "VERIFIED",
    stage_count: stagesList.length,
  };

  return {
    trace_id: traceId,
    timestamp,
    stages: stagesList,
    summary,
    measured_timings: obj.measured_timings as TraceData["measured_timings"],
    pillar_status: obj.pillar_status as TraceData["pillar_status"],
    fusion_decomposition: obj.fusion_decomposition as TraceData["fusion_decomposition"],
  };
}

function transformHistoryEntry(entry: AnalysisHistoryEntry): TraceData {
  const res = entry.result;
  const measured = res.measured_timings;
  const pStatus = res.pillar_status;
  const totalDuration = measured?.total_latency_ms ?? res.processing_time_ms ?? res.latency_ms ?? null;

  const p1Available = pStatus?.p1_available ?? (res.pillar_scores?.retrieval !== null && res.pillar_scores?.retrieval !== undefined);
  const p2Available = pStatus?.p2_available ?? (res.pillar_scores?.confidence !== null && res.pillar_scores?.confidence !== undefined);
  const p3Available = pStatus?.p3_available ?? (res.pillar_scores?.consistency !== null && res.pillar_scores?.consistency !== undefined);

  const stages: TraceStage[] = [
    {
      name: "Pillar 1 — Evidence Support",
      duration_ms: measured?.p1_latency_ms ?? null,
      status: p1Available ? "completed" : "failed",
      details: {
        factual_error: res.pillar_scores?.retrieval ?? res.pillar_scores?.pillar1_factual_error ?? null,
        evidence_count: res.evidence?.length || 0,
        root_cause: res.root_cause_classification ?? "VERIFIED",
      },
    },
    {
      name: "Pillar 2 — Model Uncertainty",
      duration_ms: measured?.p2_latency_ms ?? null,
      status: p2Available ? "completed" : "unavailable",
      details: p2Available ? {
        confidence_gap: res.pillar_scores?.confidence ?? null,
      } : { reason: "Token uncertainty analysis unavailable (no logprobs)" },
    },
    {
      name: "Pillar 3 — Generation Consistency",
      duration_ms: measured?.p3_latency_ms ?? null,
      status: p3Available ? "completed" : "unavailable",
      details: p3Available ? {
        consistency: res.pillar_scores?.consistency ?? null,
      } : { reason: "Consistency analysis unavailable" },
    },
    {
      name: "Adaptive Hybrid Fusion",
      duration_ms: measured?.fusion_latency_ms ?? null,
      status: pStatus?.fusion_status === "failed" ? "failed" : "completed",
      details: res.fusion_decomposition ? {
        mode: res.fusion_decomposition.fusion_mode,
        uncalibrated: res.fusion_decomposition.uncalibrated_h_score,
        calibrated: res.fusion_decomposition.calibrated_h_score,
      } : undefined,
    },
  ];

  return {
    trace_id: res.trace_id || entry.id,
    timestamp: entry.timestamp,
    stages,
    summary: {
      total_duration_ms: totalDuration ?? 0,
      total_memory_mb: 0,
      final_h_score: res.overall_h_score,
      risk_level: res.risk_level,
      root_cause_classification: res.root_cause_classification || "VERIFIED",
      stage_count: stages.length,
    },
    measured_timings: measured,
    pillar_status: pStatus,
    fusion_decomposition: res.fusion_decomposition,
  };
}

function getStageColor(status: string): string {
  switch (status) {
    case "completed": case "success": return "var(--verified)";
    case "failed": return "var(--hallucination)";
    case "unavailable": case "skipped": return "var(--text-dim)";
    case "running": return "var(--ai)";
    default: return "var(--text-muted)";
  }
}

function truncate(text: string, max: number) {
  if (!text) return "";
  return text.length <= max ? text : text.slice(0, max).trimEnd() + "…";
}
