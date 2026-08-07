"use client";

import React from "react";
import { motion } from "framer-motion";
import { Database, Activity, GitBranch, Layers } from "lucide-react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";
import { PILLAR_INFO } from "@/lib/constants";
import type { PillarScores } from "@/types/hallucisense";

interface PillarVisualizationProps {
  pillars: PillarScores;
  weights?: Record<string, number>;
}

const PILLAR_ICONS = {
  retrieval: Database,
  confidence: Activity,
  consistency: GitBranch,
};

export function PillarVisualization({ pillars, weights }: PillarVisualizationProps) {
  const pillarEntries = Object.entries(pillars) as [keyof PillarScores, number][];

  // Data for Recharts Radar
  const radarData = [
    { subject: "Grounding (P1)", score: pillars.retrieval * 100, fullMark: 100 },
    { subject: "Confidence (P2)", score: pillars.confidence * 100, fullMark: 100 },
    { subject: "Consistency (P3)", score: pillars.consistency * 100, fullMark: 100 },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-400">Three-Pillar Risk & Adaptive Fusion</h3>
        {weights && (
          <div className="flex items-center gap-3 text-xs font-mono text-slate-500">
            <span>α (P1): {((weights.alpha ?? weights.retrieval ?? 0.4) * 100).toFixed(0)}%</span>
            <span>β (P2): {((weights.beta ?? weights.confidence ?? 0.3) * 100).toFixed(0)}%</span>
            <span>γ (P3): {((weights.gamma ?? weights.consistency ?? 0.3) * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {pillarEntries.map(([key, value], index) => {
          const info = PILLAR_INFO[key];
          const Icon = PILLAR_ICONS[key];
          const pct = Math.min(value * 100, 100);
          const weightKey = key === "retrieval" ? "alpha" : key === "confidence" ? "beta" : "gamma";
          const weight = weights?.[weightKey] ?? weights?.[key];

          return (
            <motion.div
              key={key}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1, duration: 0.4 }}
              className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5 space-y-4 hover:border-white/[0.12] transition-all duration-300"
            >
              {/* Header */}
              <div className="flex items-center gap-3">
                <div
                  className="flex items-center justify-center w-9 h-9 rounded-xl"
                  style={{ backgroundColor: `${info.color}15` }}
                >
                  <Icon className="w-4 h-4" style={{ color: info.color }} />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-200">{info.shortName}</p>
                  <p className="text-[10px] text-slate-500">{info.name}</p>
                </div>
              </div>

              {/* Score */}
              <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold font-mono text-white">
                  {pct.toFixed(1)}
                </span>
                <span className="text-xs text-slate-500">/ 100 Risk</span>
              </div>

              {/* Progress Bar */}
              <div className="h-2 rounded-full bg-white/[0.06] overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: info.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.8, delay: 0.2 + index * 0.1, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>

              {/* Weight */}
              {weight !== undefined && (
                <div className="flex items-center justify-between pt-1">
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                    Adaptive Weight
                  </span>
                  <span className="text-xs font-mono text-slate-400">
                    {(weight * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Radar Chart Component */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
        className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5 flex flex-col md:flex-row items-center justify-between gap-6"
      >
        <div className="space-y-2 max-w-sm">
          <div className="flex items-center gap-2 text-blue-400 text-xs font-semibold uppercase tracking-wider">
            <Layers className="w-4 h-4" />
            Radar Spectrum Analysis
          </div>
          <h4 className="text-base font-bold text-white">Pillar Disagreement & Risk Balance</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Multi-dimensional risk radar profile. Disagreement between Pillar 1 (Grounding) and Pillar 2 (Predictive Confidence) triggers higher epistemic uncertainty penalties in adaptive fusion.
          </p>
        </div>

        <div className="w-full md:w-64 h-52 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.08)" />
              <PolarAngleAxis dataKey="subject" stroke="#94A3B8" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="rgba(255,255,255,0.1)" tick={false} />
              <Radar
                name="Risk Profile"
                dataKey="score"
                stroke="#3B82F6"
                fill="#2563EB"
                fillOpacity={0.4}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </div>
  );
}
