"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown,
  AlertTriangle,
  Shield,
  Minus,
  Info,
  Activity,
  FlaskConical,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { LocalAttribution, LocalAttributionFeature } from "@/types/hallucisense";
import type { SemanticGrounding } from "@/types/verification-types";

interface LocalAttributionPanelProps {
  attribution?: LocalAttribution | null;
  semanticGrounding?: SemanticGrounding | null;
  className?: string;
}

// ─── Feature name → human-readable label ──────────────────────────────────
const FEATURE_LABELS: Record<string, { label: string; group: string; description: string }> = {
  p1_mean_entailment:              { label: "Mean Entailment",           group: "Pillar 1 — Evidence",      description: "Average NLI entailment score across retrieved evidence passages" },
  p1_max_entailment:               { label: "Max Entailment",            group: "Pillar 1 — Evidence",      description: "Highest NLI entailment score from any single evidence passage" },
  p1_mean_contradiction:           { label: "Mean Contradiction",        group: "Pillar 1 — Evidence",      description: "Average NLI contradiction score from retrieved evidence" },
  p1_min_support_margin:           { label: "Min Support Margin",        group: "Pillar 1 — Evidence",      description: "Minimum margin between entailment and contradiction across claims" },
  p1_num_claims:                   { label: "Claim Count (P1)",          group: "Pillar 1 — Evidence",      description: "Number of atomic claims extracted for Pillar 1 evidence retrieval" },
  p2_max_pairwise_contradiction:   { label: "Max Pairwise Contradiction", group: "Pillar 2 — Consistency", description: "Highest contradiction score between any pair of claims" },
  p2_mean_pairwise_contradiction:  { label: "Mean Pairwise Contradiction",group: "Pillar 2 — Consistency", description: "Average pairwise NLI contradiction across all claim pairs" },
  p2_max_pairwise_similarity:      { label: "Max Pairwise Similarity",   group: "Pillar 2 — Consistency",  description: "Maximum semantic similarity between any two claims" },
  p2_fraction_contradictory_pairs: { label: "Contradictory Pair Fraction",group: "Pillar 2 — Consistency", description: "Fraction of claim pairs assessed as contradictory" },
  p2_num_claims:                   { label: "Claim Count (P2)",          group: "Pillar 2 — Consistency",  description: "Number of atomic claims for Pillar 2 pairwise analysis" },
  prob_p1:                         { label: "Pillar 1 Risk",             group: "Fusion Signals",           description: "Calibrated hallucination probability from Pillar 1 logistic model" },
  prob_p2:                         { label: "Pillar 2 Risk",             group: "Fusion Signals",           description: "Calibrated hallucination probability from Pillar 2 logistic model" },
  logit_p1:                        { label: "Pillar 1 Log-odds",         group: "Fusion Signals",           description: "Log-odds transform of Pillar 1 probability" },
  logit_p2:                        { label: "Pillar 2 Log-odds",         group: "Fusion Signals",           description: "Log-odds transform of Pillar 2 probability" },
  prob_disagreement_abs:           { label: "Pillar Disagreement",       group: "Meta Signals",             description: "Absolute difference between Pillar 1 and Pillar 2 probabilities" },
  prob_mean:                       { label: "Pillar Mean",               group: "Meta Signals",             description: "Average of Pillar 1 and Pillar 2 probabilities" },
  prob_max:                        { label: "Pillar Max",                group: "Meta Signals",             description: "Higher of the two pillar probabilities" },
  prob_min:                        { label: "Pillar Min",                group: "Meta Signals",             description: "Lower of the two pillar probabilities" },
  prob_ratio:                      { label: "Pillar Ratio",              group: "Meta Signals",             description: "Ratio of Pillar 1 to Pillar 2 probability (with epsilon clip)" },
};

function getLabel(name: string) {
  return FEATURE_LABELS[name]?.label ?? name;
}
function getDescription(name: string) {
  return FEATURE_LABELS[name]?.description ?? "";
}

// ─── Attribution bar ───────────────────────────────────────────────────────
function AttributionBar({
  feature,
  maxAbsAttribution,
}: {
  feature: LocalAttributionFeature;
  maxAbsAttribution: number;
}) {
  const pct = maxAbsAttribution > 0
    ? Math.min(100, Math.abs(feature.attribution) / maxAbsAttribution * 100)
    : 0;

  const isRisk = feature.direction === "hallucination_risk";
  const isProtective = feature.direction === "protective";

  const barColor = isRisk
    ? "bg-[var(--hallucination)]"
    : isProtective
    ? "bg-[var(--verified)]"
    : "bg-[var(--border)]";

  const textColor = isRisk
    ? "text-[var(--hallucination)]"
    : isProtective
    ? "text-[var(--verified)]"
    : "text-[var(--text-muted)]";

  const DirectionIcon = isRisk ? AlertTriangle : isProtective ? Shield : Minus;

  return (
    <div
      className="group flex items-center gap-3 py-2.5 px-3 rounded-[var(--radius-md)] hover:bg-[var(--surface-hover)] transition-colors"
      title={getDescription(feature.feature_name)}
    >
      {/* Direction icon */}
      <DirectionIcon
        className={cn("w-3.5 h-3.5 flex-shrink-0", textColor)}
        aria-label={feature.direction}
      />

      {/* Feature label */}
      <span className="text-[12px] text-[var(--text-secondary)] w-[170px] flex-shrink-0 truncate">
        {getLabel(feature.feature_name)}
      </span>

      {/* Attribution bar */}
      <div className="flex-1 h-1.5 bg-[var(--surface)] rounded-full overflow-hidden">
        <motion.div
          className={cn("h-full rounded-full", barColor)}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>

      {/* Attribution value */}
      <span className={cn("text-[11px] font-mono font-semibold w-[60px] text-right flex-shrink-0", textColor)}>
        {feature.attribution >= 0 ? "+" : ""}{(feature.attribution * 100).toFixed(2)}%
      </span>

      {/* Actual value */}
      <span className="text-[10px] text-[var(--text-dim)] font-mono w-[56px] text-right flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
        {feature.value.toFixed(4)}
      </span>
    </div>
  );
}

// ─── Probability gauge ─────────────────────────────────────────────────────
function ProbabilityDisplay({
  label,
  value,
  threshold,
  highlight,
}: {
  label: string;
  value: number;
  threshold?: number;
  highlight?: boolean;
}) {
  const pct = Math.round(value * 100);
  const isAboveThreshold = threshold !== undefined && value >= threshold;
  const color = isAboveThreshold ? "var(--hallucination)" : "var(--verified)";

  return (
    <div className={cn(
      "flex flex-col items-center p-3 rounded-[var(--radius-md)]",
      highlight ? "bg-[var(--surface)] border border-[var(--border)]" : "bg-[var(--bg-surface)]"
    )}>
      <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">{label}</span>
      <span
        className="text-2xl font-bold font-mono"
        style={{ color: highlight ? color : "var(--text-muted)" }}
      >
        {pct}%
      </span>
      {threshold !== undefined && (
        <span className="text-[10px] text-[var(--text-dim)] mt-0.5">
          τ = {Math.round(threshold * 100)}%
        </span>
      )}
    </div>
  );
}

// ─── Main panel ───────────────────────────────────────────────────────────
export function LocalAttributionPanel({ attribution, semanticGrounding, className }: LocalAttributionPanelProps) {
  const [showFullTable, setShowFullTable] = useState(false);
  const [showCaveat, setShowCaveat] = useState(false);

  if (!attribution) {
    if (semanticGrounding && semanticGrounding.claims && semanticGrounding.claims.length > 0) {
      return (
        <Card className={cn("bg-[var(--surface)] border-[var(--border)]", className)}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <Shield className="w-4 h-4 text-teal-400" />
              Evidence Grounding Trace
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {semanticGrounding.claims.map((c) => (
              <div key={c.claim_id} className="p-2.5 rounded bg-[var(--bg-surface)] border border-[var(--border)] text-xs space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-[var(--text-primary)]">&ldquo;{c.claim_text}&rdquo;</span>
                  <Badge variant="outline" className="text-[10px]">{c.primary_status}</Badge>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      );
    }
    return null;
  }

  const allFeatures = [...attribution.features].sort(
    (a, b) => Math.abs(b.attribution) - Math.abs(a.attribution)
  );
  const maxAbsAttribution = Math.max(...allFeatures.map((f) => Math.abs(f.attribution)));
  const showInteractionGap = Math.abs(attribution.interaction_gap) > 0.01;

  const isHallucinated = attribution.original_probability >= attribution.threshold;
  const marginPct = Math.abs(attribution.decision_margin * 100).toFixed(1);
  const marginDir = attribution.decision_margin >= 0 ? "above" : "below";

  const topDrivers = [
    ...attribution.top_hallucination_drivers,
    ...attribution.top_protective_drivers,
  ].sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution));

  return (
    <Card className={cn("bg-[var(--bg-surface)] border-[var(--border)]", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[var(--text-muted)]" />
            <CardTitle className="text-sm font-semibold text-[var(--text-primary)]">
              Local Counterfactual Attribution
            </CardTitle>
            <Badge
              variant="outline"
              className="text-[10px] font-mono px-1.5 py-0 border-[var(--border)] text-[var(--text-dim)]"
            >
              19 features · exact
            </Badge>
          </div>
          <span className="text-[11px] text-[var(--text-dim)] font-mono">
            {attribution.inference_count} model evaluations
          </span>
        </div>
        <p className="text-[12px] text-[var(--text-muted)] mt-1">
          Quantifies how each feature coordinate moves P(H) relative to its training-median baseline.
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* ─── Evidence Grounding Trace (Phase 39) ─── */}
        {semanticGrounding && semanticGrounding.claims && semanticGrounding.claims.length > 0 && (
          <div className="p-3.5 rounded-[var(--radius-md)] bg-[var(--surface)] border border-[var(--border)] space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
                <Shield className="w-3.5 h-3.5 text-teal-400" />
                Claim ↔ Evidence Grounding Trace
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-teal-950/60 text-teal-300 border border-teal-800/40 font-mono">
                {semanticGrounding.shadow_only ? "Shadow NLI Diagnostic" : "Active NLI Grounding"}
              </span>
            </div>

            <div className="space-y-2.5">
              {semanticGrounding.claims.map((c) => (
                <div key={c.claim_id} className="p-2.5 rounded bg-[var(--bg-surface)] border border-[var(--border)] text-xs space-y-1.5">
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-medium text-[var(--text-primary)]">Claim {c.claim_id + 1}: &ldquo;{c.claim_text}&rdquo;</span>
                    <span className={cn(
                      "px-2 py-0.5 rounded text-[10px] font-semibold tracking-wide uppercase flex-shrink-0",
                      c.primary_status === "contradiction" ? "bg-red-950/80 text-red-400 border border-red-800" :
                      c.primary_status === "entailment" ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800" :
                      c.primary_status === "neutral" ? "bg-amber-950/80 text-amber-400 border border-amber-800" :
                      "bg-zinc-900 text-zinc-400 border border-zinc-800"
                    )}>
                      {c.primary_status}
                    </span>
                  </div>

                  {c.evidence_details && c.evidence_details.length > 0 ? (
                    <div className="space-y-1 text-[11px] text-[var(--text-muted)] pl-2 border-l-2 border-[var(--border)]">
                      {c.evidence_details.slice(0, 2).map((ev, eIdx) => (
                        <div key={eIdx} className="space-y-0.5">
                          <p className="italic text-[var(--text-dim)]">&ldquo;{ev.snippet.length > 140 ? ev.snippet.slice(0, 140) + '...' : ev.snippet}&rdquo;</p>
                          <div className="flex items-center gap-3 text-[10px] font-mono text-[var(--text-dim)]">
                            <span>Source: {ev.title}</span>
                            <span className="text-emerald-400">Ent: {(ev.entailment * 100).toFixed(0)}%</span>
                            <span className="text-red-400">Con: {(ev.contradiction * 100).toFixed(0)}%</span>
                            <span className="text-amber-400">Neu: {(ev.neutral * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[11px] text-[var(--text-dim)] italic">No direct reference evidence retrieved for this claim.</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── A: Probability Summary ─────────────────────────────────────── */}
        <div className="grid grid-cols-3 gap-2">
          <ProbabilityDisplay
            label="Baseline P(H)"
            value={attribution.baseline_probability}
          />
          <ProbabilityDisplay
            label="Prediction P(H)"
            value={attribution.original_probability}
            threshold={attribution.threshold}
            highlight
          />
          <div className="flex flex-col items-center justify-center p-3 rounded-[var(--radius-md)] bg-[var(--bg-surface)]">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">
              Margin
            </span>
            <span className={cn(
              "text-lg font-bold font-mono",
              isHallucinated ? "text-[var(--hallucination)]" : "text-[var(--verified)]"
            )}>
              {attribution.decision_margin >= 0 ? "+" : ""}{(attribution.decision_margin * 100).toFixed(1)}%
            </span>
            <span className="text-[10px] text-[var(--text-dim)] mt-0.5">
              {marginPct}% {marginDir} τ
            </span>
          </div>
        </div>

        {/* ─── B: Primary Driver ─────────────────────────────────────────── */}
        {topDrivers.length > 0 && (
          <div className="p-3 rounded-[var(--radius-md)] bg-[var(--surface)] border border-[var(--border)]">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] block mb-2">
              Primary Decision Driver
            </span>
            {(() => {
              const primary = topDrivers[0];
              const isRisk = primary.direction === "hallucination_risk";
              return (
                <div className="flex items-start gap-2">
                  {isRisk
                    ? <AlertTriangle className="w-4 h-4 text-[var(--hallucination)] flex-shrink-0 mt-0.5" />
                    : <Shield className="w-4 h-4 text-[var(--verified)] flex-shrink-0 mt-0.5" />
                  }
                  <div>
                    <p className="text-[13px] font-semibold text-[var(--text-primary)]">
                      {getLabel(primary.feature_name)}
                    </p>
                    <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                      {getDescription(primary.feature_name)}.
                      {" "}Attribution:{" "}
                      <span className={cn(
                        "font-mono font-semibold",
                        isRisk ? "text-[var(--hallucination)]" : "text-[var(--verified)]"
                      )}>
                        {primary.attribution >= 0 ? "+" : ""}{(primary.attribution * 100).toFixed(2)}%
                      </span>
                      {" "}(value: <span className="font-mono">{primary.value.toFixed(4)}</span>, baseline: <span className="font-mono">{primary.baseline.toFixed(4)}</span>)
                    </p>
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* ─── C & D: Top Drivers ────────────────────────────────────────── */}
        <div className="space-y-1">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              Feature Contributions
            </span>
            <span className="text-[10px] text-[var(--text-dim)]">
              — hover row for actual value
            </span>
          </div>
          {topDrivers.map((f) => (
            <AttributionBar
              key={f.feature_name}
              feature={f}
              maxAbsAttribution={maxAbsAttribution}
            />
          ))}
        </div>

        {/* ─── E: Full 19-Feature Table ───────────────────────────────────── */}
        <div>
          <button
            onClick={() => setShowFullTable(!showFullTable)}
            className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors cursor-pointer"
          >
            <ChevronDown className={cn("w-3 h-3 transition-transform", showFullTable && "rotate-180")} />
            {showFullTable ? "Hide" : "Show"} all {attribution.feature_count} features
          </button>
          <AnimatePresence>
            {showFullTable && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-3 border border-[var(--border)] rounded-[var(--radius-md)] overflow-hidden">
                  <table className="w-full text-[11px]">
                    <thead>
                      <tr className="border-b border-[var(--border)] bg-[var(--surface)]">
                        <th className="text-left py-2 px-3 text-[var(--text-muted)] font-semibold">#</th>
                        <th className="text-left py-2 px-3 text-[var(--text-muted)] font-semibold">Feature</th>
                        <th className="text-right py-2 px-3 text-[var(--text-muted)] font-semibold">Value</th>
                        <th className="text-right py-2 px-3 text-[var(--text-muted)] font-semibold">Baseline</th>
                        <th className="text-right py-2 px-3 text-[var(--text-muted)] font-semibold">Attribution</th>
                        <th className="text-center py-2 px-3 text-[var(--text-muted)] font-semibold">Role</th>
                      </tr>
                    </thead>
                    <tbody>
                      {attribution.features.map((f) => {
                        const isRisk = f.direction === "hallucination_risk";
                        const isProtective = f.direction === "protective";
                        return (
                          <tr
                            key={f.feature_name}
                            className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--surface-hover)] transition-colors"
                          >
                            <td className="py-1.5 px-3 font-mono text-[var(--text-dim)]">{f.index}</td>
                            <td className="py-1.5 px-3 text-[var(--text-secondary)]" title={getDescription(f.feature_name)}>
                              {getLabel(f.feature_name)}
                            </td>
                            <td className="py-1.5 px-3 text-right font-mono text-[var(--text-primary)]">
                              {f.value.toFixed(5)}
                            </td>
                            <td className="py-1.5 px-3 text-right font-mono text-[var(--text-dim)]">
                              {f.baseline.toFixed(5)}
                            </td>
                            <td className={cn(
                              "py-1.5 px-3 text-right font-mono font-semibold",
                              isRisk ? "text-[var(--hallucination)]"
                                : isProtective ? "text-[var(--verified)]"
                                : "text-[var(--text-muted)]"
                            )}>
                              {f.attribution >= 0 ? "+" : ""}{(f.attribution * 100).toFixed(3)}%
                            </td>
                            <td className="py-1.5 px-3 text-center">
                              <span className={cn(
                                "px-1.5 py-0.5 rounded text-[10px] font-medium",
                                isRisk ? "bg-[var(--hallucination-soft)] text-[var(--hallucination)]"
                                  : isProtective ? "bg-[var(--verified-soft)] text-[var(--verified)]"
                                  : "text-[var(--text-dim)]"
                              )}>
                                {f.direction === "hallucination_risk" ? "Risk ↑" : f.direction === "protective" ? "Safe ↓" : "—"}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ─── G: Interaction Gap ────────────────────────────────────────── */}
        {showInteractionGap && (
          <div className="p-3 rounded-[var(--radius-md)] bg-[var(--surface)] border border-[var(--border-hover)]">
            <div className="flex items-start gap-2">
              <Info className="w-3.5 h-3.5 text-[var(--text-dim)] flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-[11px] font-semibold text-[var(--text-muted)] mb-0.5">
                  Nonlinear Interaction Gap: {attribution.interaction_gap >= 0 ? "+" : ""}{(attribution.interaction_gap * 100).toFixed(2)}%
                </p>
                <p className="text-[11px] text-[var(--text-dim)] leading-relaxed">
                  {attribution.interaction_gap_explanation}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ─── F: Scientific Caveat ──────────────────────────────────────── */}
        <div>
          <button
            onClick={() => setShowCaveat(!showCaveat)}
            className="flex items-center gap-1.5 text-[11px] text-[var(--text-dim)] hover:text-[var(--text-muted)] transition-colors cursor-pointer"
          >
            <FlaskConical className="w-3 h-3" />
            Scientific note on attribution method
            <ChevronDown className={cn("w-3 h-3 transition-transform", showCaveat && "rotate-180")} />
          </button>
          <AnimatePresence>
            {showCaveat && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-2 p-3 rounded-[var(--radius-md)] bg-[var(--surface)] border border-[var(--border)]">
                  <p className="text-[11px] text-[var(--text-dim)] leading-relaxed">
                    <strong className="text-[var(--text-muted)]">Method:</strong> Local Counterfactual Attribution.
                    For each of the {attribution.feature_count} features, the classifier is re-evaluated with that
                    feature replaced by its training-median value. The reported attribution equals the resulting
                    change in P(H). This is NOT SHAP — SHAP requires marginalising over all feature coalitions
                    via Shapley values. {attribution.scientific_caveat}
                  </p>
                  <p className="text-[11px] text-[var(--text-dim)] leading-relaxed mt-1.5">
                    <strong className="text-[var(--text-muted)]">Baseline:</strong> {attribution.baseline_type.replace(/_/g, " ")}{" "}
                    (N=58,002 development samples). Model evaluations per explanation: {attribution.inference_count}.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

      </CardContent>
    </Card>
  );
}
