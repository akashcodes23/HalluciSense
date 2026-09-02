"use client";

import React from "react";
import { motion } from "framer-motion";
import { Database, Activity, GitBranch } from "lucide-react";
import type { PillarScores } from "@/types/hallucisense";

interface PillarVisualizationProps {
  pillars: PillarScores;
  weights?: Record<string, number>;
}

export function PillarVisualization({ pillars, weights }: PillarVisualizationProps) {
  const p1 = (pillars.pillar1_factual_error ?? pillars.retrieval ?? 0);
  const p2 = (pillars.pillar2_confidence_gap ?? pillars.confidence ?? 0);
  const p3 = (pillars.pillar3_consistency_failure ?? pillars.consistency ?? 0);

  const effectiveWeights = pillars.effective_weights || {
    alpha: weights?.alpha ?? 0.45,
    beta: weights?.beta ?? 0.30,
    gamma: weights?.gamma ?? 0.25,
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-300">Three-Pillar Risk & Fusion Contributions</h3>
        <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
          <span>α (P1): {(effectiveWeights.alpha * 100).toFixed(0)}%</span>
          <span>β (P2): {(effectiveWeights.beta * 100).toFixed(0)}%</span>
          <span>γ (P3): {(effectiveWeights.gamma * 100).toFixed(0)}%</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Pillar 1 */}
        <PillarCard
          icon={<Database className="w-4 h-4 text-teal-400" />}
          title="Pillar 1 — Evidence Grounding (FE)"
          score={p1}
          weight={effectiveWeights.alpha}
          description="Hybrid BM25 + FAISS retrieval and cross-encoder NLI entailment scoring."
          colorClass="bg-teal-500"
          borderClass="border-teal-500/20"
        />

        {/* Pillar 2 */}
        <PillarCard
          icon={<Activity className="w-4 h-4 text-blue-400" />}
          title="Pillar 2 — Confidence Gap (CG)"
          score={p2}
          weight={effectiveWeights.beta}
          description="Token log-probability distribution and Shannon entropy H(p) uncertainty analysis."
          colorClass="bg-blue-500"
          borderClass="border-blue-500/20"
        />

        {/* Pillar 3 */}
        <PillarCard
          icon={<GitBranch className="w-4 h-4 text-amber-400" />}
          title="Pillar 3 — Consistency Failure (CF)"
          score={p3}
          weight={effectiveWeights.gamma}
          description="Multi-sample semantic consistency and cross-generation contradiction analysis."
          colorClass="bg-amber-500"
          borderClass="border-amber-500/20"
        />
      </div>
    </div>
  );
}

function PillarCard({
  icon,
  title,
  score,
  weight,
  description,
  colorClass = "bg-teal-500",
  borderClass = "border-white/[0.06]",
}: {
  icon: React.ReactNode;
  title: string;
  score: number | null;
  weight: number;
  description: string;
  colorClass?: string;
  borderClass?: string;
}) {
  const isAvailable = score != null && !isNaN(score);
  const pct = isAvailable ? Math.min(Math.max(score * 100, 0), 100) : 0;

  return (
    <div className={`p-4 rounded-xl border ${borderClass} bg-[#0b1220] space-y-3`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-xs font-semibold text-white">{title}</span>
        </div>
        <span className="text-xs font-mono text-slate-500">Weight: {(weight * 100).toFixed(0)}%</span>
      </div>

      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="text-slate-400">Risk Metric</span>
          <span className="font-bold text-white">{isAvailable ? `${pct.toFixed(1)}%` : "N/A (Protected)"}</span>
        </div>
        <div className="h-2 rounded-full bg-white/[0.06] overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${colorClass}`}
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      <p className="text-[11px] text-slate-500 leading-tight">{description}</p>
    </div>
  );
}
