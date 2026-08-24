"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ExternalLink, CheckCircle2, XCircle, MinusCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { SentenceScore, EvidenceItem } from "@/types/hallucisense";

interface ClaimAnalysisCardProps {
  claim: SentenceScore;
  index: number;
}

export function ClaimAnalysisCard({ claim, index }: ClaimAnalysisCardProps) {
  const [expanded, setExpanded] = useState(false);

  const text = claim.sentence_text || claim.text || "";
  const score = claim.h_score ?? claim.score ?? null;
  const riskLevel = claim.risk_level;
  const evidence = claim.evidence_matched || [];

  const riskColor = getRiskTokenColor(riskLevel);
  const scorePercent = score !== null && score !== undefined ? Math.round(score * 100) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.25 }}
      className="rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-surface)] overflow-hidden"
    >
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start gap-3 p-4 text-left hover:bg-[var(--surface-hover)] transition-colors cursor-pointer"
        aria-expanded={expanded}
      >
        {/* Risk Indicator */}
        <div className="shrink-0 mt-0.5">
          <span
            className="flex items-center justify-center w-6 h-6 rounded-full text-[10px] font-bold"
            style={{ backgroundColor: riskColor + "18", color: riskColor }}
          >
            {index + 1}
          </span>
        </div>

        {/* Claim Text */}
        <div className="flex-1 min-w-0">
          <p className="text-[13px] text-[var(--text-primary)] leading-relaxed">
            {text}
          </p>
          <div className="flex flex-wrap items-center gap-2 mt-2">
            <Badge
              variant={getBadgeVariant(riskLevel)}
              size="sm"
            >
              {getStatusIcon(riskLevel)}
              {getRiskLabel(riskLevel)}
            </Badge>
            {claim.epistemic_category && (
              <Badge variant="ai" size="sm">{claim.epistemic_category.replace(/_/g, " ")}</Badge>
            )}
            {evidence.length > 0 && (
              <Badge variant="evidence" size="sm">{evidence.length} source{evidence.length !== 1 ? "s" : ""}</Badge>
            )}
          </div>
        </div>

        {/* Score */}
        <div className="shrink-0 text-right">
          <p className="text-lg font-bold font-mono" style={{ color: riskColor }}>
            {scorePercent !== null ? `${scorePercent}%` : "—"}
          </p>
          <p className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider">H-Score</p>
        </div>

        <ChevronDown className={cn("w-4 h-4 text-[var(--text-dim)] shrink-0 mt-1 transition-transform", expanded && "rotate-180")} />
      </button>

      {/* Expanded Content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 border-t border-[var(--border)] pt-3 space-y-4">
              {/* NLI Signals */}
              {(claim.nli_entailment_prob !== undefined || claim.nli_contradiction_prob !== undefined) && (
                <div>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">
                    NLI Signals
                  </h4>
                  <div className="grid grid-cols-3 gap-2">
                    <SignalCell label="Entailment" value={claim.nli_entailment_prob} format="pct" />
                    <SignalCell label="Contradiction" value={claim.nli_contradiction_prob} format="pct" />
                    <SignalCell label="Neutral" value={claim.nli_neutral_prob} format="pct" />
                  </div>
                </div>
              )}

              {/* Reasoning */}
              {claim.reasoning_summary && (
                <div>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">
                    Reasoning
                  </h4>
                  <p className="text-[13px] text-[var(--text-secondary)] leading-relaxed">
                    {claim.reasoning_summary}
                  </p>
                </div>
              )}

              {/* Evidence */}
              {evidence.length > 0 && (
                <div>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">
                    Evidence ({evidence.length})
                  </h4>
                  <div className="space-y-2">
                    {evidence.map((ev, i) => (
                      <EvidenceRow key={i} evidence={ev} index={i} />
                    ))}
                  </div>
                </div>
              )}

              {/* Temporal Anchor */}
              {claim.temporal_anchor && (
                <div>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">
                    Temporal Analysis
                  </h4>
                  <div className="flex gap-4 text-[13px]">
                    <span className="text-[var(--text-muted)]">
                      Asserted: <span className="text-[var(--text-secondary)] font-mono">{claim.temporal_anchor.asserted_year ?? "—"}</span>
                    </span>
                    <span className="text-[var(--text-muted)]">
                      Evidence: <span className="text-[var(--text-secondary)] font-mono">{claim.temporal_anchor.evidence_year ?? "—"}</span>
                    </span>
                    <span className="text-[var(--text-muted)]">
                      Compatible:{" "}
                      <span className={claim.temporal_anchor.is_compatible ? "text-[var(--verified)]" : "text-[var(--hallucination)]"}>
                        {claim.temporal_anchor.is_compatible ? "Yes" : "No"}
                      </span>
                    </span>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function SignalCell({ label, value, format }: { label: string; value?: number | null; format?: "pct" | "raw" }) {
  const display = value !== null && value !== undefined
    ? format === "pct" ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
    : "—";
  return (
    <div className="rounded-[var(--radius)] bg-[var(--surface)] p-2 text-center">
      <p className="text-[10px] text-[var(--text-dim)] uppercase tracking-wider mb-0.5">{label}</p>
      <p className="text-sm font-mono font-medium text-[var(--text-secondary)]">{display}</p>
    </div>
  );
}

function EvidenceRow({ evidence, index }: { evidence: EvidenceItem; index: number }) {
  const similarity = evidence.similarity_score ?? evidence.score;
  return (
    <div className="rounded-[var(--radius)] bg-[var(--surface)] p-3">
      <div className="flex items-start justify-between gap-2 mb-1">
        <span className="text-[11px] font-semibold text-[var(--evidence)]">
          Source {index + 1}
          {evidence.source_name ? `: ${evidence.source_name}` : evidence.source ? `: ${evidence.source}` : ""}
        </span>
        {similarity !== undefined && similarity !== null && (
          <Badge variant="evidence" size="sm">
            {(similarity * 100).toFixed(0)}% match
          </Badge>
        )}
      </div>
      <p className="text-[12px] text-[var(--text-secondary)] leading-relaxed line-clamp-3">
        {evidence.snippet}
      </p>
      {(evidence.source_url) && (
        <a
          href={evidence.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-[11px] text-[var(--evidence)] hover:underline mt-1"
        >
          <ExternalLink className="w-3 h-3" /> View source
        </a>
      )}
    </div>
  );
}

function getRiskTokenColor(level: string): string {
  switch (level) {
    case "VERIFIED": return "var(--verified)";
    case "NEEDS_VERIFICATION": case "MODERATE_RISK": return "var(--warning)";
    case "LIKELY_HALLUCINATED": return "var(--hallucination)";
    default: return "var(--text-muted)";
  }
}

function getBadgeVariant(level: string): "verified" | "hallucination" | "warning" | "outline" {
  switch (level) {
    case "VERIFIED": return "verified";
    case "LIKELY_HALLUCINATED": return "hallucination";
    case "NEEDS_VERIFICATION": case "MODERATE_RISK": return "warning";
    default: return "outline";
  }
}

function getStatusIcon(level: string) {
  switch (level) {
    case "VERIFIED": return <CheckCircle2 className="w-3 h-3" />;
    case "LIKELY_HALLUCINATED": return <XCircle className="w-3 h-3" />;
    case "NEEDS_VERIFICATION": case "MODERATE_RISK": return <MinusCircle className="w-3 h-3" />;
    default: return null;
  }
}

function getRiskLabel(level: string): string {
  switch (level) {
    case "VERIFIED": return "Verified";
    case "NEEDS_VERIFICATION": return "Needs Review";
    case "MODERATE_RISK": return "Moderate Risk";
    case "LIKELY_HALLUCINATED": return "Hallucinated";
    default: return level;
  }
}
