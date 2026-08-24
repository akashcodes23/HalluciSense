"use client";

import React, { useState, useMemo } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  Search,
  Filter,
  XCircle,
  Clock,
  GitBranch,
  ChevronRight,
  ShieldCheck,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { Button } from "@/components/ui/button";
import { NoErrors } from "@/components/ui/EmptyState";
import { useAnalysisStore } from "@/store/analysis-store";
import { formatScore, formatTimestamp } from "@/lib/format";
import { cn } from "@/lib/utils";
import Link from "next/link";

const FILTER_TABS = [
  { id: "all", label: "All" },
  { id: "hallucination", label: "Hallucinations" },
  { id: "numerical", label: "Numerical" },
  { id: "negation", label: "Negation" },
  { id: "causal", label: "Causal" },
  { id: "retrieval", label: "Retrieval" },
  { id: "nli", label: "NLI" },
];

export default function ErrorFeedPage() {
  const history = useAnalysisStore((s) => s.history);
  const setSelectedTraceId = useAnalysisStore((s) => s.setSelectedTraceId);

  const [activeFilter, setActiveFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedError, setSelectedError] = useState<string | null>(null);

  // Filter to only errors (non-verified results)
  const errors = useMemo(() => {
    return history
      .filter((h) => {
        const rl = h.result.risk_level;
        if (rl === "VERIFIED") return false;

        // Category filter
        if (activeFilter !== "all") {
          const rc = (h.result.root_cause_classification || "").toLowerCase();
          if (activeFilter === "hallucination" && rl !== "LIKELY_HALLUCINATED") return false;
          if (activeFilter === "numerical" && !rc.includes("numer") && !rc.includes("unit")) return false;
          if (activeFilter === "negation" && !rc.includes("negat")) return false;
          if (activeFilter === "causal" && !rc.includes("causal")) return false;
          if (activeFilter === "retrieval" && !rc.includes("retrieval")) return false;
          if (activeFilter === "nli" && !rc.includes("nli") && !rc.includes("entail")) return false;
        }

        // Search filter
        if (searchQuery) {
          const q = searchQuery.toLowerCase();
          const text = `${h.query} ${h.response} ${h.result.root_cause_classification || ""}`.toLowerCase();
          if (!text.includes(q)) return false;
        }

        return true;
      })
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [history, activeFilter, searchQuery]);

  const selectedEntry = selectedError ? history.find((h) => h.id === selectedError) : null;

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Error List ──────────────────────────────────────────────── */}
      <div className={cn("flex-1 min-w-0 flex flex-col", selectedEntry && "hidden lg:flex")}>
        {/* Header */}
        <div className="p-5 md:p-6 pb-0 space-y-4">
          <div>
            <h1 className="text-heading-lg text-[var(--text-primary)]">Error Feed</h1>
            <p className="text-label-md text-[var(--text-muted)] mt-1">
              Hallucination detections, verification failures, and anomalies
            </p>
          </div>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-dim)]" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search errors…"
              className={cn(
                "w-full pl-9 pr-3 py-2 rounded-[var(--radius)]",
                "bg-[var(--bg-surface)] border border-[var(--border)]",
                "text-sm text-[var(--text-primary)] placeholder:text-[var(--text-dim)]",
                "focus:outline-none focus:border-[var(--primary)]",
                "transition-all duration-150"
              )}
            />
          </div>

          {/* Filter Tabs */}
          <div className="flex gap-1 overflow-x-auto pb-1">
            {FILTER_TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveFilter(tab.id)}
                className={cn(
                  "px-3 py-1.5 rounded-[var(--radius-sm)] text-[12px] font-medium whitespace-nowrap",
                  "transition-all duration-150 cursor-pointer",
                  activeFilter === tab.id
                    ? "bg-[var(--primary-soft)] text-[var(--primary)] border border-[var(--ai-border)]"
                    : "text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] border border-transparent"
                )}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Error List */}
        <div className="flex-1 overflow-y-auto px-5 md:px-6 py-3 space-y-1">
          {errors.length > 0 ? (
            errors.map((entry) => (
              <button
                key={entry.id}
                onClick={() => setSelectedError(entry.id)}
                className={cn(
                  "w-full flex items-center gap-3 p-3 rounded-[var(--radius-md)] text-left",
                  "transition-all duration-100 cursor-pointer",
                  selectedError === entry.id
                    ? "bg-[var(--primary-soft)] border border-[var(--ai-border)]"
                    : "hover:bg-[var(--surface-hover)] border border-transparent"
                )}
              >
                <StatusBadge status={entry.result.risk_level} size="sm" showIcon={true} />
                <div className="flex-1 min-w-0">
                  <p className="text-[13px] text-[var(--text-primary)] truncate">
                    {truncate(entry.query || entry.response, 60)}
                  </p>
                  <div className="flex items-center gap-2 mt-0.5">
                    {entry.result.root_cause_classification && entry.result.root_cause_classification !== "NONE" && (
                      <span className="text-[10px] text-[var(--text-dim)] font-mono">
                        {entry.result.root_cause_classification.replace(/_/g, " ")}
                      </span>
                    )}
                    <span className="text-[10px] text-[var(--text-dim)]">
                      {formatTimestamp(entry.timestamp)}
                    </span>
                  </div>
                </div>
                <span className="text-sm font-mono font-bold text-[var(--hallucination)] shrink-0">
                  {formatScore(entry.result.overall_h_score)}%
                </span>
                <ChevronRight className="w-4 h-4 text-[var(--text-dim)] shrink-0" />
              </button>
            ))
          ) : (
            <NoErrors />
          )}
        </div>
      </div>

      {/* ── Error Detail Panel ──────────────────────────────────────── */}
      {selectedEntry && (
        <motion.div
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-full lg:w-[480px] lg:border-l border-[var(--border)] bg-[var(--bg-surface)] overflow-y-auto shrink-0"
        >
          <div className="p-5 space-y-4">
            {/* Close on mobile */}
            <div className="flex items-center justify-between lg:hidden">
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">Error Detail</h3>
              <Button variant="ghost" size="icon-sm" onClick={() => setSelectedError(null)}>
                <XCircle className="w-4 h-4" />
              </Button>
            </div>

            {/* Status */}
            <StatusBadge status={selectedEntry.result.risk_level} size="lg" />

            {/* H-Score */}
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">H-Score</p>
              <p className="text-3xl font-bold font-mono text-[var(--hallucination)]">
                {formatScore(selectedEntry.result.overall_h_score)}%
              </p>
            </div>

            {/* Root Cause */}
            {selectedEntry.result.root_cause_classification && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">Root Cause</p>
                <Badge variant="hallucination" size="lg">
                  {selectedEntry.result.root_cause_classification.replace(/_/g, " ")}
                </Badge>
              </div>
            )}

            {/* Query */}
            {selectedEntry.query && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">Original Query</p>
                <p className="text-[13px] text-[var(--text-secondary)]">{selectedEntry.query}</p>
              </div>
            )}

            {/* Response */}
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">Response Analyzed</p>
              <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed line-clamp-6">{selectedEntry.response}</p>
            </div>

            {/* Pillar Scores */}
            {selectedEntry.result.pillar_scores && (
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">Pillar Signals</p>
                <div className="space-y-1.5">
                  <PillarRow label="Evidence Grounding" value={selectedEntry.result.pillar_scores.retrieval ?? selectedEntry.result.pillar_scores.pillar1_factual_error} />
                  <PillarRow label="Confidence" value={selectedEntry.result.pillar_scores.confidence ?? selectedEntry.result.pillar_scores.pillar2_confidence_gap} />
                  <PillarRow label="Consistency" value={selectedEntry.result.pillar_scores.consistency ?? selectedEntry.result.pillar_scores.pillar3_consistency_failure} />
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2 pt-2">
              {selectedEntry.result.trace_id && (
                <Link href={`/traces?id=${selectedEntry.result.trace_id}`}>
                  <Button variant="outline" size="sm">
                    <GitBranch className="w-3.5 h-3.5" /> Open Trace
                  </Button>
                </Link>
              )}
              <Button variant="outline" size="sm" disabled>
                Create Evaluation Case
              </Button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

function PillarRow({ label, value }: { label: string; value?: number | null }) {
  const isAvailable = value !== null && value !== undefined;
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-[12px] text-[var(--text-secondary)]">{label}</span>
      <span className={cn("text-[13px] font-mono font-medium", isAvailable ? "text-[var(--text-primary)]" : "text-[var(--text-dim)]")}>
        {isAvailable ? `${(value * 100).toFixed(1)}%` : "—"}
      </span>
    </div>
  );
}

function truncate(text: string, max: number) {
  if (!text) return "";
  return text.length <= max ? text : text.slice(0, max).trimEnd() + "…";
}
