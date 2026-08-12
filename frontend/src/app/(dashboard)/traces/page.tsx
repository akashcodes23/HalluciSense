"use client";

import React, { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import {
  GitBranch,
  Clock,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  Search,
  RefreshCw,
  Loader2,
  ListFilter,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { GlassCard } from "@/components/ui/card";
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
      setTraceIdInput(selectedTraceId); // eslint-disable-line
      setSearchId(selectedTraceId); // eslint-disable-line
      setSelectedTraceId(null); // Clear selected trace to avoid loop // eslint-disable-line
    }
  }, [selectedTraceId, setSelectedTraceId]);

  const { data: latestTrace, isLoading: latestLoading, refetch } = useLatestTrace();
  const { data: searchedTrace, isLoading: searchLoading } = useDebugTrace(searchId);

  const isLoading = latestLoading || searchLoading;

  // Local helper to convert history entry to TraceData structure
  const transformHistoryEntry = (entry: AnalysisHistoryEntry): TraceData => {
    const duration = entry.result.processing_time_ms || entry.result.latency_ms || 1200;
    const stages = [
      {
        name: "Pillar 1 — Evidence Grounding",
        duration_ms: duration * 0.45,
        status: "completed",
        details: { 
          factual_error: entry.result.pillar_scores?.retrieval ?? 0,
          root_cause: entry.result.root_cause_classification,
          evidence_count: entry.result.evidence?.length || 0
        }
      },
      {
        name: "Pillar 2 — Confidence Estimation",
        duration_ms: duration * 0.3,
        status: "completed",
        details: { 
          entropy: entry.result.pillar_scores?.confidence ?? 0,
          confidence_score: entry.result.confidence ?? 0.5
        }
      },
      {
        name: "Pillar 3 — Consistency Reasoning",
        duration_ms: duration * 0.25,
        status: "completed",
        details: { 
          consistency: entry.result.pillar_scores?.consistency ?? 0,
          failure_taxonomy: entry.result.failure_taxonomy
        }
      },
      {
        name: "Adaptive Fusion Engine",
        duration_ms: 0.05,
        status: "completed",
        details: { 
          final_h_score: entry.result.overall_h_score,
          risk_level: entry.result.risk_level
        }
      }
    ];

    return {
      trace_id: entry.result.trace_id || entry.id,
      timestamp: entry.timestamp,
      stages: stages as TraceStage[],
      summary: {
        total_duration_ms: duration,
        total_memory_mb: 256.0,
        final_h_score: entry.result.overall_h_score,
        risk_level: entry.result.risk_level,
        root_cause_classification: entry.result.root_cause_classification || "VERIFIED",
        stage_count: stages.length
      }
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

  const displayTrace = localTrace || searchedTrace || latestTrace || fallbackLatestLocalTrace;

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
        <div className="flex items-center justify-between border-b border-white/[0.06] pb-6">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-indigo-600 shadow-[0_0_24px_rgba(168,85,247,0.25)]">
              <GitBranch className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">Pipeline Traces</h1>
              <p className="text-sm text-slate-500">Execution timeline & diagnostic logs</p>
            </div>
          </div>

          <Button
            variant="secondary"
            size="sm"
            onClick={() => refetch()}
            className="flex items-center gap-1.5 border border-white/10 hover:border-white/20 transition-all cursor-pointer"
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
              className="pl-10 bg-slate-950/40 border-white/10 text-white rounded-xl focus-visible:ring-indigo-500"
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            />
          </div>
          <Button
            onClick={handleSearch}
            disabled={!traceIdInput.trim() || isLoading}
            className="bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl px-5 cursor-pointer disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Search"}
          </Button>
        </div>

        {/* ── Main Layout Split Grid ────────────────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Left Sidebar: Local Traces List */}
          <div className="md:col-span-1 space-y-4">
            <GlassCard className="p-4 border-white/[0.08] bg-[#070b13]/60 rounded-xl space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider border-b border-white/5 pb-2">
                <ListFilter className="w-3.5 h-3.5 text-indigo-400" />
                <span>Session History ({history.length})</span>
              </div>

              {history.length === 0 ? (
                <div className="text-center py-6 text-slate-600 text-xs leading-relaxed">
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
                            ? "bg-indigo-500/10 border-indigo-500/30 text-white shadow-sm"
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
            </GlassCard>
          </div>

          {/* Right Column: Active Trace details */}
          <div className="md:col-span-3 space-y-6">
            {isLoading ? (
              <div className="flex items-center justify-center py-24 bg-[#070b13]/40 border border-white/5 rounded-2xl">
                <div className="text-center space-y-3">
                  <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mx-auto" />
                  <p className="text-xs text-slate-500">Retrieving diagnostic timeline data...</p>
                </div>
              </div>
            ) : displayTrace ? (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Trace Summary Card */}
                <GlassCard className="p-6 border-white/[0.08] bg-[#070b13]/80 rounded-2xl">
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 px-2 py-0.5 rounded font-mono font-semibold uppercase tracking-wider">
                          Trace Log
                        </span>
                        <span className="text-xs font-mono text-slate-500">{displayTrace.trace_id}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <h2 className="text-2xl font-bold text-white tracking-tight font-sans">
                          H-Score: {displayTrace.summary ? (displayTrace.summary.final_h_score * 100).toFixed(1) : "0.0"}%
                        </h2>
                        <Badge
                          variant={
                            displayTrace.summary?.risk_level === "VERIFIED"
                              ? "verified"
                              : displayTrace.summary?.risk_level === "LIKELY_HALLUCINATED"
                              ? "danger"
                              : "warning"
                          }
                          className="px-2.5 py-0.5 rounded-lg text-xs"
                        >
                          {getRiskLabel(displayTrace.summary?.risk_level || "VERIFIED")}
                        </Badge>
                      </div>
                    </div>
                    <div className="text-right text-xs text-slate-500 space-y-1 font-mono">
                      <p className="font-sans">{formatTimestamp(displayTrace.timestamp)}</p>
                      <p>RSS: {displayTrace.summary?.total_memory_mb ? `${displayTrace.summary.total_memory_mb.toFixed(0)} MB` : "N/A"}</p>
                      <p className="text-slate-400">Verdict: {displayTrace.summary?.root_cause_classification || "VERIFIED"}</p>
                    </div>
                  </div>
                </GlassCard>

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
              <div className="flex flex-col items-center justify-center py-20 text-center bg-[#070b13]/40 border border-dashed border-white/10 rounded-2xl p-6">
                <div className="w-14 h-14 rounded-2xl bg-white/[0.02] border border-white/5 flex items-center justify-center mb-4">
                  <FileText className="w-6 h-6 text-slate-500" />
                </div>
                <h3 className="text-sm font-semibold text-slate-300 mb-1">No Active Trace Selected</h3>
                <p className="text-slate-500 text-xs max-w-sm leading-relaxed mb-6">
                  Select a past execution trace from your session history sidebar on the left, or query a trace ID in the search bar.
                </p>
                <Button
                  onClick={() => window.location.href = "/verify"}
                  className="bg-white/5 hover:bg-white/10 text-white rounded-xl border border-white/10 text-xs px-4"
                >
                  Go to Verification Workspace
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function TraceStageRow({ stage, index }: { stage: TraceStage; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const isSuccess = stage.status === "completed" || stage.status === "success" || !stage.status;

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
            : "bg-red-500 border-red-500/30"
        )}
        style={{ transform: "translateX(-4.5px)" }}
      />

      {/* Card */}
      <div className="ml-6">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full text-left rounded-xl border border-white/[0.06] bg-[#070b13]/40 px-4 py-3 hover:border-white/[0.12] hover:bg-[#070b13]/60 transition-all cursor-pointer"
          aria-expanded={expanded}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {isSuccess ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <XCircle className="w-4 h-4 text-red-400" />
              )}
              <span className="text-sm font-medium text-slate-200">{stage.name}</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1 text-xs text-slate-500 font-mono">
                <Clock className="w-3 h-3" />
                {formatLatency(stage.duration_ms)}
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
