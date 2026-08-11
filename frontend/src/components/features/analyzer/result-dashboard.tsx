"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Clock,
  Cpu,
  Database,
  AlertTriangle,
  Info,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { GlassCard } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ScoreGauge } from "@/components/features/analyzer/score-gauge";
import { PillarVisualization } from "@/components/features/analyzer/pillar-visualization";
import { TokenHeatmap } from "@/components/features/heatmap/token-heatmap";
import { EvidenceExplorer } from "@/components/features/evidence/evidence-explorer";
import { formatLatency, getRiskColor, getRiskLabel } from "@/lib/format";
import type { AnalysisResponse, ExplainResponse } from "@/types/hallucisense";

interface ResultDashboardProps {
  result: AnalysisResponse;
  explain: ExplainResponse | null;
}

const riskIcon = (level: string) => {
  switch (level) {
    case "VERIFIED": return <ShieldCheck className="w-5 h-5" />;
    case "LOW_RISK": return <ShieldCheck className="w-5 h-5" />;
    case "MODERATE_RISK":
    case "NEEDS_VERIFICATION": return <ShieldAlert className="w-5 h-5" />;
    case "LIKELY_HALLUCINATED": return <ShieldX className="w-5 h-5" />;
    default: return <Info className="w-5 h-5" />;
  }
};

const riskBadgeVariant = (level: string) => {
  switch (level) {
    case "VERIFIED": return "verified" as const;
    case "LOW_RISK": return "info" as const;
    case "MODERATE_RISK":
    case "NEEDS_VERIFICATION": return "warning" as const;
    case "LIKELY_HALLUCINATED": return "danger" as const;
    default: return "default" as const;
  }
};

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};

const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export function ResultDashboard({ result, explain }: ResultDashboardProps) {
  const confidencePct = ((result.confidence ?? (1 - result.overall_h_score)) * 100).toFixed(1);
  const latencyMs = result.latency_ms ?? result.processing_time_ms ?? 0;
  const traceIdStr = result.trace_id ? result.trace_id.slice(0, 12) : "LOCAL_EXEC";

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="space-y-6">
      {/* ── Score Overview Row ──────────────────────────────────────── */}
      <motion.div variants={item}>
        <GlassCard className="p-6">
          <div className="flex flex-col md:flex-row items-center gap-8">
            {/* Score Gauge */}
            <ScoreGauge score={result.overall_h_score} riskLevel={result.risk_level} />

            {/* Score Details */}
            <div className="flex-1 space-y-4 text-center md:text-left">
              <div>
                <Badge variant={riskBadgeVariant(result.risk_level)} className="mb-2">
                  {riskIcon(result.risk_level)}
                  {getRiskLabel(result.risk_level)}
                </Badge>
                <h2 className="text-2xl font-bold text-white tracking-tight">
                  Hallucination Score: {(result.overall_h_score * 100).toFixed(1)}%
                </h2>
                <p className="text-sm text-slate-400 mt-1">
                  System confidence: {confidencePct}%
                </p>
              </div>

              {/* Meta Stats */}
              <div className="flex flex-wrap gap-4 justify-center md:justify-start">
                <MetaStat icon={<Clock className="w-3.5 h-3.5" />} label="Latency" value={formatLatency(latencyMs)} />
                <MetaStat icon={<Cpu className="w-3.5 h-3.5" />} label="Version" value={result.version || "1.0 Production"} />
                <MetaStat icon={<AlertTriangle className="w-3.5 h-3.5" />} label="Root Cause" value={result.root_cause_classification || "None"} />
                <MetaStat icon={<Database className="w-3.5 h-3.5" />} label="Trace" value={traceIdStr} />
              </div>
            </div>
          </div>
        </GlassCard>
      </motion.div>

      {/* ── Three Pillars ──────────────────────────────────────────── */}
      <motion.div variants={item}>
        <PillarVisualization
          pillars={result.pillar_scores}
          weights={explain?.adaptive_weights}
        />
      </motion.div>

      {/* ── Tabbed Detail Views ────────────────────────────────────── */}
      <motion.div variants={item}>
        <Tabs defaultValue="sentences">
          <TabsList>
            <TabsTrigger value="sentences">Sentence Claims</TabsTrigger>
            <TabsTrigger value="heatmap">Token Heatmap</TabsTrigger>
            <TabsTrigger value="evidence">Evidence Citations</TabsTrigger>
            {explain && <TabsTrigger value="reasoning">Reasoning</TabsTrigger>}
          </TabsList>

          <TabsContent value="sentences">
            <SentenceList sentences={result.sentence_scores} />
          </TabsContent>

          <TabsContent value="heatmap">
            <TokenHeatmap tokens={result.token_heatmap || []} />
          </TabsContent>

          <TabsContent value="evidence">
            <EvidenceExplorer
              evidence={result.evidence || []}
              explainEvidence={explain?.retrieved_evidence}
              supporting={explain?.supporting_passages}
              contradicting={explain?.contradiction_evidence}
            />
          </TabsContent>

          {explain && (
            <TabsContent value="reasoning">
              <ReasoningChain
                chain={explain.reasoning_chain || []}
                explanation={explain.confidence_explanation || explain.explanation_markdown || ""}
                fusion={explain.fusion_contribution || {}}
              />
            </TabsContent>
          )}
        </Tabs>
      </motion.div>
    </motion.div>
  );
}

/* ── Sub-components ──────────────────────────────────────────────────────── */

function MetaStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06]">
      <span className="text-slate-500">{icon}</span>
      <span className="text-xs text-slate-500">{label}</span>
      <span className="text-xs font-medium text-slate-300 font-mono">{value}</span>
    </div>
  );
}

function SentenceList({ sentences }: { sentences: AnalysisResponse["sentence_scores"] }) {
  if (!sentences || !sentences.length) {
    return (
      <div className="text-center py-12 text-slate-500 text-sm">
        No sentence-level scores available.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {sentences.map((s, i) => {
        const textStr = s.sentence_text || s.text || "";
        const scoreVal = s.h_score ?? s.score ?? 0;
        return (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05 }}
            className="flex items-start gap-3 px-4 py-3 rounded-xl border border-white/[0.06] bg-white/[0.02]"
          >
            <div
              className="w-2 h-2 rounded-full mt-2 shrink-0"
              style={{ backgroundColor: getRiskColor(s.risk_level) }}
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-slate-200">{textStr}</p>
              <p className="text-xs text-slate-500 mt-1">
                Score: {(scoreVal * 100).toFixed(1)}% • {getRiskLabel(s.risk_level)}
              </p>
            </div>
            <Badge variant={riskBadgeVariant(s.risk_level)} className="shrink-0 text-[10px]">
              {(scoreVal * 100).toFixed(0)}%
            </Badge>
          </motion.div>
        );
      })}
    </div>
  );
}

function ReasoningChain({
  chain,
  explanation,
  fusion,
}: {
  chain: string[];
  explanation: string;
  fusion: Record<string, number>;
}) {
  return (
    <div className="space-y-6">
      {chain.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-slate-400 mb-3">Reasoning Chain</h4>
          {chain.map((step, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              className="flex gap-3 px-4 py-3 rounded-xl border border-white/[0.06] bg-white/[0.02]"
            >
              <span className="text-xs font-mono text-blue-400 mt-0.5 shrink-0">
                {String(i + 1).padStart(2, "0")}
              </span>
              <p className="text-sm text-slate-300">{step}</p>
            </motion.div>
          ))}
        </div>
      )}

      {explanation && (
        <div className="px-4 py-3 rounded-xl border border-white/[0.06] bg-white/[0.02]">
          <h4 className="text-sm font-medium text-slate-400 mb-2">Confidence Explanation</h4>
          <p className="text-sm text-slate-300">{explanation}</p>
        </div>
      )}

      {Object.keys(fusion).length > 0 && (
        <div className="px-4 py-3 rounded-xl border border-white/[0.06] bg-white/[0.02]">
          <h4 className="text-sm font-medium text-slate-400 mb-3">Fusion Contributions</h4>
          <div className="space-y-2">
            {Object.entries(fusion).map(([key, val]) => (
              <div key={key} className="flex items-center gap-3">
                <span className="text-xs text-slate-500 w-24 truncate font-mono">{key}</span>
                <div className="flex-1 h-2 rounded-full bg-white/[0.06] overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-blue-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(val * 100, 100)}%` }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                  />
                </div>
                <span className="text-xs font-mono text-slate-400 w-12 text-right">
                  {(val * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
