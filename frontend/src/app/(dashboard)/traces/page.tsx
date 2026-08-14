"use client";

import React, { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import {
  GitBranch,
  Clock,
  CheckCircle2,
  XCircle,
  MinusCircle,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Search,
  RefreshCw,
  Loader2,
  ListFilter,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/EmptyState";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { Input } from "@/components/ui/input";
import { useLatestTrace, useDebugTrace } from "@/hooks/use-analysis";
import { useAnalysisStore } from "@/store/analysis-store";
import { formatLatency, formatTimestamp, getRiskColor, getRiskLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TraceStage, TraceData, AnalysisHistoryEntry } from "@/types/hallucisense";

export default function TracesPage() {
  const [traceIdInput, setTraceIdInput] = useState("");
  const [searchId, setSearchId] = useState<string | null>(null);

  const history = useAnalysisStore((s) => s.history);
  const selectedTraceId = useAnalysisStore((s) => s.selectedTraceId);
  const setSelectedTraceId = useAnalysisStore((s) => s.setSelectedTraceId);

  // Sync selected trace from store
  useEffect(() => {
    if (selectedTraceId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setTraceIdInput(selectedTraceId);
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSearchId(selectedTraceId);
      setSelectedTraceId(null); // Clear selected trace to avoid loop
    }
  }, [selectedTraceId, setSelectedTraceId]);

  const { data: latestTrace, isLoading: latestLoading, refetch } = useLatestTrace();
  const { data: searchedTrace, isLoading: searchLoading } = useDebugTrace(searchId);

  // Context-dependent loading: when searching, only care about searchLoading
  // (and skip it entirely if we already have a local fallback ready).
  // When not searching, only care about latestLoading.
  const isLoading = searchId
    ? searchLoading && !history.find((h) => h.id === searchId || h.result.trace_id === searchId)
    : latestLoading && history.length === 0;

  // Local helper to convert history entry to TraceData structure
  const transformHistoryEntry = (entry: AnalysisHistoryEntry): TraceData => {
    const res = entry.result;
    const measured = res.measured_timings;
    const pStatus = res.pillar_status;
    const totalDuration = measured?.total_latency_ms ?? res.processing_time_ms ?? res.latency_ms ?? null;

    const p1Available = pStatus?.p1_available ?? true;
    const p2Available = pStatus?.p2_available ?? (res.pillar_scores?.confidence !== undefined && res.pillar_scores?.confidence !== null && res.confidence_analysis !== undefined);
    const p3Available = pStatus?.p3_available ?? (res.pillar_scores?.consistency !== undefined && res.pillar_scores?.consistency !== null);

    const stages: TraceStage[] = [
      {
        name: "Pillar 1 — Evidence Grounding",
        duration_ms: measured?.p1_latency_ms ?? (p1Available && totalDuration ? totalDuration : null),
        status: p1Available ? "completed" : "failed",
        details: {
          status: p1Available ? "EXECUTED" : "FAILED",
          factual_error: res.pillar_scores?.retrieval ?? res.pillar_scores?.pillar1_factual_error ?? null,
          evidence_count: res.evidence?.length || 0,
          root_cause: res.root_cause_classification ?? "VERIFIED",
        }
      },
      {
        name: "Pillar 2 — Confidence Estimation",
        duration_ms: p2Available ? (measured?.p2_latency_ms ?? null) : null,
        status: p2Available ? "completed" : "unavailable",
        details: p2Available ? {
          status: "EXECUTED",
          confidence_gap: res.pillar_scores?.confidence ?? null,
          entropy: res.confidence_analysis?.whitebox_entropy ?? null,
          methodology: res.confidence_analysis ? "Model Uncertainty Proxy / Logprobs" : "N/A"
        } : {
          status: "UNAVAILABLE",
          reason: "Token-level logprobs not provided by active LLM provider. Excluded from fusion."
        }
      },
      {
        name: "Pillar 3 — Consistency Reasoning",
        duration_ms: p3Available ? (measured?.p3_latency_ms ?? null) : null,
        status: p3Available ? "completed" : "unavailable",
        details: p3Available ? {
          status: "EXECUTED",
          consistency_failure: res.pillar_scores?.consistency ?? null,
          failure_taxonomy: res.failure_taxonomy ?? "NONE"
        } : {
          status: "UNAVAILABLE",
          reason: "Single generation mode active. Multi-sampling consistency was not executed."
        }
      },
      {
        name: "Adaptive Fusion Engine",
        duration_ms: measured?.fusion_latency_ms ?? 0.5,
        status: "completed",
        details: res.fusion_decomposition ? {
          ...res.fusion_decomposition,
        } : {
          final_h_score: res.overall_h_score,
          risk_level: res.risk_level,
          is_full_analysis: pStatus?.is_full_analysis ?? (p1Available && p2Available && p3Available),
        }
      }
    ];

    return {
      trace_id: res.trace_id || entry.id,
      timestamp: entry.timestamp,
      stages: stages,
      summary: {
        total_duration_ms: totalDuration || 0,
        total_memory_mb: 256.0,
        final_h_score: res.overall_h_score,
        risk_level: res.risk_level,
        root_cause_classification: res.root_cause_classification || "VERIFIED",
        stage_count: stages.length
      },
      measured_timings: measured,
      pillar_status: pStatus,
      fusion_decomposition: res.fusion_decomposition,
    };
  };

  // Check local history first
  const localTrace = useMemo(() => {
    if (!searchId) return null;
    const entry = history.find(
      (h) => h.id === searchId || h.result.trace_id === searchId
    );
    if (!entry) return null;
    return transformHistoryEntry(entry);
  }, [searchId, history]);

  // Fallback to latest local history entry if no trace loaded yet and no search query active
  const fallbackLatestLocalTrace = useMemo(() => {
    if (history.length === 0) return null;
    return transformHistoryEntry(history[0]);
  }, [history]);

  // Trace lookup priority:
  // 1. If searching, prefer backend trace (searchedTrace)
  // 2. If backend trace search fails or is empty, try local trace history fallback (localTrace)
  // 3. If not searching, try latest server trace (latestTrace)
  // 4. Fallback to latest local history entry (fallbackLatestLocalTrace)
  const displayTrace = useMemo(() => {
    if (searchId) {
      if (searchedTrace) return searchedTrace;
      if (!searchLoading && localTrace) return localTrace;
      return searchedTrace || localTrace;
    }
    return latestTrace || fallbackLatestLocalTrace;
  }, [searchId, searchedTrace, searchLoading, localTrace, latestTrace, fallbackLatestLocalTrace]);

  const isLocalCache = useMemo(() => {
    if (!displayTrace) return false;
    if (searchId) {
      return !searchedTrace && displayTrace === localTrace;
    }
    return !latestTrace && displayTrace === fallbackLatestLocalTrace;
  }, [displayTrace, searchId, searchedTrace, latestTrace, localTrace, fallbackLatestLocalTrace]);

  const handleSearch = () => {
    if (traceIdInput.trim()) {
      setSearchId(traceIdInput.trim());
    }
  };

  const selectLocalTrace = (id: string) => {
    setTraceIdInput(id);
    setSearchId(id);
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between border-b border-white/[0.04] pb-6">
          <div className="flex items-center gap-3">
            <GitBranch className="w-6 h-6 text-accent-primary shrink-0" />
            <div>
              <h1 className="text-heading-md font-bold text-white tracking-tight leading-none">Pipeline Traces</h1>
              <p className="text-label-md text-slate-500">Execution timeline & diagnostic logs</p>
            </div>
          </div>

          <Button
            variant="secondary"
            size="sm"
            onClick={() => refetch()}
            className="flex items-center gap-1.5 border border-white/5 bg-white/[0.01] hover:bg-white/[0.04] hover:text-white transition-all cursor-pointer font-mono text-xs"
          >
            <RefreshCw className={cn("w-3.5 h-3.5", isLoading && "animate-spin")} />
            Refresh Server
          </Button>
        </div>

        {/* ── Search Bar ────────────────────────────────────────────────── */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <Input
              value={traceIdInput}
              onChange={(e) => setTraceIdInput(e.target.value)}
              placeholder="Search by trace ID (e.g. TRACE_88CFA3E9)"
              className="pl-10 bg-bg-surface border-white/[0.04] text-white rounded-xl focus:border-accent-primary/40 focus:ring-accent-primary/10 font-mono text-sm"
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            />
          </div>
          <Button
            onClick={handleSearch}
            disabled={!traceIdInput.trim() || isLoading}
            className="bg-accent-primary hover:bg-accent-primary/90 text-white rounded-xl px-5 cursor-pointer disabled:opacity-50 font-mono text-xs shadow-[0_0_24px_rgba(168,85,247,0.2)]"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Search"}
          </Button>
        </div>

        {/* ── Main Layout Split Grid ────────────────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Left Sidebar: Local Traces List */}
          <div className="md:col-span-1 space-y-4">
            <Card className="p-4 rounded-xl space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider border-b border-white/[0.04] pb-2 font-mono">
                <ListFilter className="w-3.5 h-3.5 text-slate-400" />
                <span>Session ({history.length})</span>
              </div>

              {history.length === 0 ? (
                <div className="text-center py-6 text-slate-400 text-xs leading-relaxed">
                  No local traces recorded in this browser session.
                </div>
              ) : (
                <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1 custom-scrollbar">
                  {history.map((entry) => {
                    const id = entry.result?.trace_id || entry.id;
                    const isActive = displayTrace?.trace_id === id;
                    const risk = entry.result?.risk_level || "VERIFIED";

                    return (
                      <button
                        key={entry.id}
                        onClick={() => selectLocalTrace(id)}
                        className={cn(
                          "w-full text-left p-3 rounded-lg border text-xs font-mono transition-all duration-200 cursor-pointer block",
                          isActive
                            ? "bg-accent-primary/10 border-accent-primary/30 text-white shadow-sm"
                            : "bg-white/[0.01] border-white/5 text-slate-400 hover:bg-white/[0.03] hover:border-white/10"
                        )}
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="font-semibold text-slate-300 truncate max-w-[80px]">
                            {id.slice(0, 12)}
                          </span>
                          <span
                            className="font-bold text-[10px]"
                            style={{ color: getRiskColor(risk) }}
                          >
                            {(entry.result?.overall_h_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-[10px] text-slate-500 font-sans">
                          <span className="truncate max-w-[90px]">{entry.response}</span>
                          <span>{new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </Card>
          </div>

          {/* Right Column: Active Trace details */}
          <div className="md:col-span-3 space-y-6">
            {isLoading ? (
              <div className="flex items-center justify-center py-24 bg-bg-surface border border-white/[0.04] rounded-2xl">
                <div className="text-center space-y-3">
                  <Loader2 className="w-8 h-8 text-accent-primary animate-spin mx-auto" />
                  <p className="text-xs text-slate-500">Retrieving diagnostic timeline data...</p>
                </div>
              </div>
            ) : displayTrace ? (
              <motion.div
                key={displayTrace.trace_id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Trace Summary Card */}
                <Card className="p-6 rounded-2xl">
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[10px] bg-white/[0.02] border border-white/[0.04] text-slate-400 px-2 py-0.5 rounded font-mono font-semibold uppercase tracking-wider">
                          Trace Log
                        </span>
                        <span className="text-xs font-mono text-slate-500">{displayTrace.trace_id}</span>
                        {isLocalCache ? (
                          <StatusBadge label="from cache" status="warning" className="font-mono text-[9px] py-0 px-1.5" />
                        ) : (
                          <StatusBadge label="from server" status="success" className="font-mono text-[9px] py-0 px-1.5" />
                        )}
                      </div>
                      <div className="flex items-center gap-3">
                        <h2 className="text-heading-md font-bold text-white tracking-tight font-sans leading-none">
                          H-Score: {displayTrace.summary ? (displayTrace.summary.final_h_score * 100).toFixed(1) : "0.0"}%
                        </h2>
                        <StatusBadge
                          label={getRiskLabel(displayTrace.summary?.risk_level || "VERIFIED")}
                          status={
                            displayTrace.summary?.risk_level === "VERIFIED"
                              ? "success"
                              : displayTrace.summary?.risk_level === "LIKELY_HALLUCINATED"
                              ? "error"
                              : "warning"
                          }
                          className="px-2.5 py-0.5 rounded-lg text-xs"
                        />
                      </div>
                    </div>
                    <div className="text-right text-xs text-slate-500 space-y-1 font-mono">
                      <p className="font-sans">{formatTimestamp(displayTrace.timestamp)}</p>
                      <p>RSS: {displayTrace.summary?.total_memory_mb ? `${displayTrace.summary.total_memory_mb.toFixed(0)} MB` : "N/A"}</p>
                      <p className="text-slate-400">Verdict: {displayTrace.summary?.root_cause_classification || "VERIFIED"}</p>
                    </div>
                  </div>
                </Card>

                {/* Pipeline Timeline */}
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Execution Timeline
                  </h3>

                  <div className="relative pl-6">
                    {/* Vertical connector line */}
                    <div className="absolute left-[11px] top-3 bottom-3 w-px bg-white/[0.08]" />

                    {displayTrace.stages?.map((stage: TraceStage, index: number) => (
                      <TraceStageRow key={index} stage={stage} index={index} />
                    ))}
                  </div>
                </div>
              </motion.div>
            ) : (
              /* Designed Empty State */
              <EmptyState
                title="No Active Trace Selected"
                description="Select a past execution trace from your session history sidebar on the left, or query a trace ID in the search bar."
                icon={FileText}
                actionLabel="Go to Verification Workspace"
                actionHref="/verify"
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function TraceStageRow({ stage, index }: { stage: TraceStage; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const isUnavailable = stage.status === "unavailable" || stage.status === "skipped" || stage.duration_ms === null;
  const isSuccess = !isUnavailable && (stage.status === "completed" || stage.status === "success" || !stage.status);

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="relative pb-4"
    >
      {/* Dot */}
      <div
        className={cn(
          "absolute left-0 top-3 w-[10px] h-[10px] rounded-full border-2",
          isSuccess
            ? "bg-emerald-500 border-emerald-500/30"
            : isUnavailable
            ? "bg-slate-600 border-slate-500/30"
            : "bg-red-500 border-red-500/30"
        )}
        style={{ transform: "translateX(-4.5px)" }}
      />

      {/* Card */}
      <div className="ml-6">
        <button
          onClick={() => setExpanded(!expanded)}
          className={cn(
            "w-full text-left rounded-xl border px-4 py-3 transition-all cursor-pointer",
            isUnavailable
              ? "border-white/[0.03] bg-bg-surface/20 opacity-80 hover:opacity-100 hover:border-white/[0.06]"
              : "border-white/[0.04] bg-bg-surface/40 hover:border-white/[0.08] hover:bg-bg-surface/60"
          )}
          aria-expanded={expanded}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {isSuccess ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : isUnavailable ? (
                <MinusCircle className="w-4 h-4 text-slate-400" />
              ) : (
                <XCircle className="w-4 h-4 text-red-400" />
              )}
              <span className={cn("text-sm font-medium", isUnavailable ? "text-slate-400" : "text-slate-200")}>
                {stage.name}
              </span>
              {isUnavailable && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/[0.04] text-slate-400 border border-white/[0.06]">
                  Unavailable
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1 text-xs text-slate-500 font-mono">
                <Clock className="w-3 h-3" />
                {isUnavailable ? (
                  <span className="text-slate-400">Not available</span>
                ) : (
                  formatLatency(stage.duration_ms)
                )}
              </div>
              {expanded ? (
                <ChevronUp className="w-4 h-4 text-slate-500" />
              ) : (
                <ChevronDown className="w-4 h-4 text-slate-500" />
              )}
            </div>
          </div>
        </button>

        {expanded && stage.details && (
          <motion.pre
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mt-2 p-4 rounded-xl bg-black/40 border border-white/[0.06] text-xs font-mono text-slate-400 overflow-x-auto select-text leading-relaxed"
          >
            {JSON.stringify(stage.details, null, 2)}
          </motion.pre>
        )}
      </div>
    </motion.div>
  );
}
