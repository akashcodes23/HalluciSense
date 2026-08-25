"use client";

import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  Search,
  Filter,
  XCircle,
  Clock,
  GitBranch,
  ChevronRight,
  ShieldCheck,
  ShieldAlert,
  Trash2,
  X,
  MessageSquare,
  CheckCircle2,
  Info,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { NoErrors } from "@/components/ui/EmptyState";
import { useAnalysisStore, type VerificationErrorEvent, type ErrorEventSource } from "@/store/analysis-store";
import { formatScore, formatTimestamp } from "@/lib/format";
import { cn } from "@/lib/utils";
import Link from "next/link";

// ─── Constants ────────────────────────────────────────────────────────────────
const RISK_FILTER_TABS = [
  { id: "all", label: "All Events" },
  { id: "LIKELY_HALLUCINATED", label: "Hallucinated" },
  { id: "MODERATE_RISK", label: "Moderate Risk" },
  { id: "NEEDS_VERIFICATION", label: "Unverified" },
  { id: "CORRECTED", label: "Corrected" },
  { id: "FAILED", label: "Failed" },
];

const SOURCE_FILTERS: { id: "all" | ErrorEventSource; label: string }[] = [
  { id: "all", label: "All Sources" },
  { id: "VERIFY", label: "Verify" },
  { id: "CHAT", label: "Chat" },
  { id: "SYSTEM", label: "System" },
];

// ─── Risk level styling ───────────────────────────────────────────────────────
function riskConfig(rl: string) {
  switch (rl) {
    case "LIKELY_HALLUCINATED":
      return { color: "text-rose-400", bg: "bg-rose-500/15", border: "border-rose-500/30", icon: <AlertTriangle className="w-3.5 h-3.5" />, label: "HALLUCINATED" };
    case "MODERATE_RISK":
      return { color: "text-amber-400", bg: "bg-amber-500/15", border: "border-amber-500/30", icon: <ShieldAlert className="w-3.5 h-3.5" />, label: "MODERATE RISK" };
    case "NEEDS_VERIFICATION":
      return { color: "text-indigo-400", bg: "bg-indigo-500/15", border: "border-indigo-500/30", icon: <Info className="w-3.5 h-3.5" />, label: "UNVERIFIED" };
    case "CORRECTED":
      return { color: "text-amber-300", bg: "bg-amber-500/10", border: "border-amber-400/30", icon: <AlertTriangle className="w-3.5 h-3.5" />, label: "CORRECTED" };
    case "VERIFIED":
      return { color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/25", icon: <CheckCircle2 className="w-3.5 h-3.5" />, label: "VERIFIED" };
    case "FAILED":
      return { color: "text-rose-300", bg: "bg-rose-900/20", border: "border-rose-500/20", icon: <XCircle className="w-3.5 h-3.5" />, label: "FAILED" };
    case "REVIEW":
      return { color: "text-sky-400", bg: "bg-sky-500/10", border: "border-sky-500/25", icon: <Info className="w-3.5 h-3.5" />, label: "REVIEW" };
    default:
      return { color: "text-slate-400", bg: "bg-slate-500/10", border: "border-slate-500/20", icon: <Info className="w-3.5 h-3.5" />, label: rl };
  }
}

// ─── Source badge ─────────────────────────────────────────────────────────────
function SourceBadge({ source }: { source: ErrorEventSource }) {
  if (source === "CHAT") {
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 shrink-0">
        <MessageSquare className="w-2.5 h-2.5" />
        CHAT
      </span>
    );
  }
  if (source === "VERIFY") {
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30 shrink-0">
        <ShieldCheck className="w-2.5 h-2.5" />
        VERIFY
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold bg-slate-500/20 text-slate-300 border border-slate-500/30 shrink-0">
      SYS
    </span>
  );
}

// ─── Risk Status Badge ────────────────────────────────────────────────────────
function RiskBadge({ risk_level, size = "sm" }: { risk_level: string; size?: "sm" | "lg" }) {
  const cfg = riskConfig(risk_level);
  const cls = size === "lg" ? "px-3 py-1 text-xs" : "px-2 py-0.5 text-[10px]";
  return (
    <span className={cn(
      "inline-flex items-center gap-1.5 font-bold rounded-lg border",
      cls, cfg.color, cfg.bg, cfg.border
    )}>
      {cfg.icon}
      {cfg.label}
    </span>
  );
}

// ─── Pillar Row ───────────────────────────────────────────────────────────────
function PillarRow({ label, value }: { label: string; value?: number | null }) {
  const ok = value !== null && value !== undefined;
  return (
    <div className="flex items-center justify-between py-1 border-b border-white/[0.04] last:border-0">
      <span className="text-[12px] text-slate-400">{label}</span>
      <span className={cn("text-[13px] font-mono font-medium", ok ? "text-slate-200" : "text-slate-600")}>
        {ok ? `${(value! * 100).toFixed(1)}%` : "—"}
      </span>
    </div>
  );
}

function truncate(text: string | undefined, max: number) {
  if (!text) return "";
  return text.length <= max ? text : text.slice(0, max).trimEnd() + "…";
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function ErrorFeedPage() {
  const errorFeed = useAnalysisStore((s) => s.errorFeed);
  const removeErrorEvent = useAnalysisStore((s) => s.removeErrorEvent);
  const clearErrorFeed = useAnalysisStore((s) => s.clearErrorFeed);

  const [activeRisk, setActiveRisk] = useState("all");
  const [activeSource, setActiveSource] = useState<"all" | ErrorEventSource>("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showVerified, setShowVerified] = useState(false);

  const filtered = useMemo(() => {
    return errorFeed
      .filter((e) => {
        // By default hide VERIFIED events unless toggle is on
        if (!showVerified && e.risk_level === "VERIFIED") return false;

        if (activeRisk !== "all" && e.risk_level !== activeRisk) return false;
        if (activeSource !== "all" && e.source !== activeSource) return false;

        if (searchQuery) {
          const q = searchQuery.toLowerCase();
          const text = `${e.query ?? ""} ${e.response} ${e.root_cause ?? ""} ${e.error_message ?? ""}`.toLowerCase();
          if (!text.includes(q)) return false;
        }
        return true;
      })
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [errorFeed, activeRisk, activeSource, searchQuery, showVerified]);

  const selectedEntry: VerificationErrorEvent | undefined = selectedId
    ? errorFeed.find((e) => e.id === selectedId)
    : undefined;

  const hallucinatedCount = errorFeed.filter((e) => e.risk_level === "LIKELY_HALLUCINATED").length;
  const failedCount = errorFeed.filter((e) => e.risk_level === "FAILED").length;
  const correctedCount = errorFeed.filter((e) => e.risk_level === "CORRECTED").length;

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Error List ──────────────────────────────────────────────────── */}
      <div className={cn("flex-1 min-w-0 flex flex-col", selectedEntry && "hidden lg:flex")}>
        {/* Header */}
        <div className="p-5 md:p-6 pb-0 space-y-4">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-heading-lg text-[var(--text-primary)]">Error Feed</h1>
              <p className="text-label-md text-[var(--text-muted)] mt-1">
                Hallucination detections, verification failures, and anomalies
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {/* Summary chips */}
              {hallucinatedCount > 0 && (
                <span className="px-2 py-1 rounded-lg text-[11px] font-bold bg-rose-500/15 text-rose-400 border border-rose-500/30">
                  {hallucinatedCount} hallucinated
                </span>
              )}
              {failedCount > 0 && (
                <span className="px-2 py-1 rounded-lg text-[11px] font-bold bg-rose-900/20 text-rose-300 border border-rose-500/20">
                  {failedCount} failed
                </span>
              )}
              {correctedCount > 0 && (
                <span className="px-2 py-1 rounded-lg text-[11px] font-bold bg-amber-500/10 text-amber-300 border border-amber-400/30">
                  {correctedCount} corrected
                </span>
              )}
              {errorFeed.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => { clearErrorFeed(); setSelectedId(null); }}
                  className="text-[var(--text-dim)] hover:text-rose-400 gap-1.5"
                  title="Clear all events"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear All
                </Button>
              )}
            </div>
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-dim)]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by query, response, or root cause…"
              className={cn(
                "w-full pl-9 pr-3 py-2 rounded-[var(--radius)]",
                "bg-[var(--bg-surface)] border border-[var(--border)]",
                "text-sm text-[var(--text-primary)] placeholder:text-[var(--text-dim)]",
                "focus:outline-none focus:border-[var(--primary)]",
                "transition-all duration-150"
              )}
            />
          </div>

          {/* Source + Risk filter row */}
          <div className="flex flex-wrap gap-2">
            {/* Source */}
            <div className="flex gap-1">
              {SOURCE_FILTERS.map((f) => (
                <button
                  key={f.id}
                  onClick={() => setActiveSource(f.id as "all" | ErrorEventSource)}
                  className={cn(
                    "px-2.5 py-1 rounded-[var(--radius-sm)] text-[11px] font-medium whitespace-nowrap",
                    "transition-all duration-150 cursor-pointer border",
                    activeSource === f.id
                      ? "bg-[var(--primary-soft)] text-[var(--primary)] border-[var(--ai-border)]"
                      : "text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] border-transparent"
                  )}
                >
                  {f.label}
                </button>
              ))}
            </div>

            <div className="h-auto w-px bg-[var(--border)]" />

            {/* Risk */}
            <div className="flex gap-1 overflow-x-auto">
              {RISK_FILTER_TABS.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveRisk(tab.id)}
                  className={cn(
                    "px-2.5 py-1 rounded-[var(--radius-sm)] text-[11px] font-medium whitespace-nowrap",
                    "transition-all duration-150 cursor-pointer border",
                    activeRisk === tab.id
                      ? "bg-[var(--primary-soft)] text-[var(--primary)] border-[var(--ai-border)]"
                      : "text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] border-transparent"
                  )}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Show verified toggle */}
            <button
              onClick={() => setShowVerified((v) => !v)}
              className={cn(
                "ml-auto px-2.5 py-1 rounded-[var(--radius-sm)] text-[11px] font-medium whitespace-nowrap",
                "transition-all duration-150 cursor-pointer border flex items-center gap-1",
                showVerified
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/25"
                  : "text-[var(--text-dim)] border-transparent hover:text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]"
              )}
            >
              <CheckCircle2 className="w-3 h-3" />
              {showVerified ? "Hiding verified" : "Show verified"}
            </button>
          </div>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto px-5 md:px-6 py-3 space-y-1">
          <AnimatePresence initial={false}>
            {filtered.length > 0 ? (
              filtered.map((entry) => {
                const cfg = riskConfig(entry.risk_level);
                return (
                  <motion.div
                    key={entry.id}
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                    transition={{ duration: 0.15 }}
                    className={cn(
                      "group flex items-center gap-3 p-3 rounded-[var(--radius-md)]",
                      "transition-all duration-100 cursor-pointer border",
                      selectedId === entry.id
                        ? "bg-[var(--primary-soft)] border-[var(--ai-border)]"
                        : "hover:bg-[var(--surface-hover)] border-transparent"
                    )}
                    onClick={() => setSelectedId(entry.id)}
                  >
                    {/* Risk icon */}
                    <span className={cn("p-1.5 rounded-lg border shrink-0", cfg.bg, cfg.border, cfg.color)}>
                      {cfg.icon}
                    </span>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <SourceBadge source={entry.source} />
                        {entry.root_cause && entry.root_cause !== "NONE" && (
                          <span className="text-[9px] font-mono text-[var(--text-dim)] truncate">
                            {entry.root_cause.replace(/_/g, " ")}
                          </span>
                        )}
                      </div>
                      <p className="text-[13px] text-[var(--text-primary)] truncate leading-tight">
                        {truncate(entry.query || entry.response, 65)}
                      </p>
                      <span className="text-[10px] text-[var(--text-dim)]">
                        {formatTimestamp(entry.timestamp)}
                      </span>
                    </div>

                    {entry.h_score !== undefined && entry.h_score !== null ? (
                      <span className={cn("text-sm font-mono font-bold shrink-0", cfg.color)}>
                        {formatScore(entry.h_score)}%
                      </span>
                    ) : (
                      <span className="text-xs text-[var(--text-dim)] shrink-0">—</span>
                    )}

                    {/* Dismiss */}
                    <button
                      onClick={(ev) => { ev.stopPropagation(); removeErrorEvent(entry.id); if (selectedId === entry.id) setSelectedId(null); }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-rose-500/20 hover:text-rose-400 text-[var(--text-dim)] transition-all cursor-pointer"
                      title="Dismiss"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>

                    <ChevronRight className="w-4 h-4 text-[var(--text-dim)] shrink-0" />
                  </motion.div>
                );
              })
            ) : (
              <NoErrors />
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Detail Panel ─────────────────────────────────────────────────── */}
      {selectedEntry && (
        <motion.div
          key={selectedEntry.id}
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-full lg:w-[480px] lg:border-l border-[var(--border)] bg-[var(--bg-surface)] overflow-y-auto shrink-0"
        >
          <div className="p-5 space-y-4">
            {/* Mobile close */}
            <div className="flex items-center justify-between lg:hidden">
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">Event Detail</h3>
              <Button variant="ghost" size="icon-sm" onClick={() => setSelectedId(null)}>
                <XCircle className="w-4 h-4" />
              </Button>
            </div>

            {/* Status */}
            <div className="flex items-center gap-2 flex-wrap">
              <RiskBadge risk_level={selectedEntry.risk_level} size="lg" />
              <SourceBadge source={selectedEntry.source} />
            </div>

            {/* H-Score */}
            {selectedEntry.h_score !== undefined && selectedEntry.h_score !== null && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">H-Score</p>
                <p className={cn("text-3xl font-bold font-mono", riskConfig(selectedEntry.risk_level).color)}>
                  {formatScore(selectedEntry.h_score)}%
                </p>
              </div>
            )}

            {/* Root Cause */}
            {selectedEntry.root_cause && selectedEntry.root_cause !== "NONE" && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">Root Cause</p>
                <Badge variant="hallucination" size="lg">
                  {selectedEntry.root_cause.replace(/_/g, " ")}
                </Badge>
              </div>
            )}

            {/* Error message */}
            {selectedEntry.error_message && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">Error</p>
                <p className="text-[13px] text-rose-300">{selectedEntry.error_message}</p>
              </div>
            )}

            {/* Query */}
            {selectedEntry.query && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">Query</p>
                <p className="text-[13px] text-[var(--text-secondary)]">{selectedEntry.query}</p>
              </div>
            )}

            {/* Response */}
            {selectedEntry.response && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">Response Analysed</p>
                <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed line-clamp-6">{selectedEntry.response}</p>
              </div>
            )}

            {/* Pillar Scores */}
            {selectedEntry.pillar_scores && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">Pillar Signals</p>
                <div className="rounded-[var(--radius-md)] border border-[var(--border)] px-3 py-1">
                  <PillarRow label="Evidence Grounding" value={selectedEntry.pillar_scores.retrieval ?? selectedEntry.pillar_scores.pillar1_factual_error} />
                  <PillarRow label="Confidence Gap" value={selectedEntry.pillar_scores.confidence ?? selectedEntry.pillar_scores.pillar2_confidence_gap} />
                  <PillarRow label="Consistency" value={selectedEntry.pillar_scores.consistency ?? selectedEntry.pillar_scores.pillar3_consistency_failure} />
                </div>
              </div>
            )}

            {/* Timestamp */}
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">Time</p>
              <p className="text-[12px] font-mono text-[var(--text-secondary)]">{new Date(selectedEntry.timestamp).toLocaleString()}</p>
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-2 flex-wrap">
              {selectedEntry.trace_id && (
                <Link href={`/traces?id=${selectedEntry.trace_id}`}>
                  <Button variant="outline" size="sm">
                    <GitBranch className="w-3.5 h-3.5" />
                    Open Trace
                  </Button>
                </Link>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={() => { removeErrorEvent(selectedEntry.id); setSelectedId(null); }}
                className="text-rose-400 hover:text-rose-300 hover:border-rose-500/40"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Dismiss
              </Button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

