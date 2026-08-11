"use client";

import React, { useState } from "react";
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
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { GlassCard } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useLatestTrace, useDebugTrace } from "@/hooks/use-analysis";
import { formatLatency, formatTimestamp, getRiskColor, getRiskLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TraceStage } from "@/types/hallucisense";

export default function TracesPage() {
  const [traceIdInput, setTraceIdInput] = useState("");
  const [searchId, setSearchId] = useState<string | null>(null);

  const { data: latestTrace, isLoading: latestLoading, refetch } = useLatestTrace();
  const { data: searchedTrace, isLoading: searchLoading } = useDebugTrace(searchId);

  const displayTrace = searchedTrace || latestTrace;
  const isLoading = latestLoading || searchLoading;

  const handleSearch = () => {
    if (traceIdInput.trim()) {
      setSearchId(traceIdInput.trim());
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-indigo-600 shadow-[0_0_24px_rgba(168,85,247,0.25)]">
              <GitBranch className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">Pipeline Traces</h1>
              <p className="text-sm text-slate-500">Execution timeline & debug data</p>
            </div>
          </div>

          <Button variant="secondary" size="sm" onClick={() => refetch()}>
            <RefreshCw className="w-4 h-4" />
            Refresh
          </Button>
        </motion.div>

        {/* ── Search ──────────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex gap-2"
        >
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <Input
              value={traceIdInput}
              onChange={(e) => setTraceIdInput(e.target.value)}
              placeholder="Search by trace ID (e.g. TRACE_88CFA3E9)"
              className="pl-10"
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            />
          </div>
          <Button variant="secondary" onClick={handleSearch} disabled={!traceIdInput.trim()}>
            Search
          </Button>
        </motion.div>

        {/* ── Loading ─────────────────────────────────────────────────── */}
        {isLoading && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
          </div>
        )}

        {/* ── Trace Display ───────────────────────────────────────────── */}
        {displayTrace && !isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="space-y-6"
          >
            {/* Summary Card */}
            <GlassCard className="p-6">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                  <p className="text-xs text-slate-500 font-mono mb-1">{displayTrace.trace_id}</p>
                  <div className="flex items-center gap-3">
                    <h2 className="text-xl font-bold text-white">
                      H-Score: {(displayTrace.summary.final_h_score * 100).toFixed(1)}%
                    </h2>
                    <Badge variant={displayTrace.summary.risk_level === "VERIFIED" ? "verified" : displayTrace.summary.risk_level === "LIKELY_HALLUCINATED" ? "danger" : "warning"}>
                      {getRiskLabel(displayTrace.summary.risk_level)}
                    </Badge>
                  </div>
                </div>
                <div className="text-right text-xs text-slate-500">
                  <p>{formatTimestamp(displayTrace.timestamp)}</p>
                  <p className="font-mono">{displayTrace.summary.root_cause_classification}</p>
                </div>
              </div>
            </GlassCard>

            {/* Pipeline Timeline */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-slate-400">Execution Timeline</h3>

              <div className="relative pl-6">
                {/* Vertical Line */}
                <div className="absolute left-[11px] top-3 bottom-3 w-px bg-white/[0.06]" />

                {displayTrace.stages.map((stage: TraceStage, index: number) => (
                  <TraceStageRow key={index} stage={stage} index={index} />
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* ── Empty State ─────────────────────────────────────────────── */}
        {!displayTrace && !isLoading && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-4">
              <GitBranch className="w-7 h-7 text-slate-600" />
            </div>
            <p className="text-slate-500 text-sm max-w-sm">
              No traces available yet. Run an analysis to generate execution traces.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function TraceStageRow({ stage, index }: { stage: TraceStage; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const isSuccess = stage.status === "completed" || stage.status === "success";

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.08 }}
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
          className="w-full text-left rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3 hover:border-white/[0.1] transition-all cursor-pointer"
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
              <div className="flex items-center gap-1 text-xs text-slate-500">
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

        {expanded && stage.output && (
          <motion.pre
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mt-2 ml-0 p-4 rounded-xl bg-black/30 border border-white/[0.06] text-xs font-mono text-slate-400 overflow-x-auto"
          >
            {JSON.stringify(stage.output, null, 2)}
          </motion.pre>
        )}
      </div>
    </motion.div>
  );
}
