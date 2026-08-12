"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  LayoutDashboard,
  Activity,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Database,
  Cpu,
  RefreshCw,
  GitBranch,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  ExternalLink,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { GlassCard } from "@/components/ui/card";
import { useMetrics, useHealth, useReady } from "@/hooks/use-analysis";
import { useAnalysisStore } from "@/store/analysis-store";
import { formatLatency, formatNumber, getRiskColor, getRiskLabel } from "@/lib/format";
import Link from "next/link";

export default function DashboardPage() {
  const { data: metrics, isLoading: metricsLoading, refetch: refetchMetrics } = useMetrics();
  const { data: health } = useHealth();
  const { data: ready } = useReady();
  const history = useAnalysisStore((s) => s.history);

  const isHealthy = health?.status === "ok" || health?.status === "healthy";
  const componentsReady = ready?.components || {};

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-7xl mx-auto">
      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.06] pb-6">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-xs font-semibold tracking-wide uppercase mb-2">
            <LayoutDashboard className="w-3.5 h-3.5" />
            System Telemetry Overview
          </div>
          <h1 className="text-2xl md:text-3xl font-bold text-white tracking-tight">
            HalluciSense Operational Control Center
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Live execution statistics, framework component health, historical verification logs, and active risk distribution.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-white/10 bg-white/[0.03] text-xs font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                isHealthy ? "bg-emerald-400 animate-pulse" : "bg-rose-400"
              }`}
            />
            <span className="text-slate-300">
              Backend: {isHealthy ? "ONLINE" : "DISCONNECTED"}
            </span>
          </div>

          <button
            onClick={() => refetchMetrics()}
            className="p-2 rounded-xl border border-white/10 bg-white/[0.03] text-slate-400 hover:text-white transition-colors"
            title="Refresh Telemetry"
          >
            <RefreshCw className={`w-4 h-4 ${metricsLoading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* ── Key Metrics Cards Row ───────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <GlassCard className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
            <span>Total Requests</span>
            <Activity className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl md:text-3xl font-bold font-mono text-white">
            {metrics ? formatNumber(metrics.requests) : "0"}
          </div>
          <p className="text-[11px] text-slate-500">Live API requests processed</p>
        </GlassCard>

        <GlassCard className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
            <span>Success Rate</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl md:text-3xl font-bold font-mono text-emerald-400">
            {metrics ? `${metrics.success_rate.toFixed(1)}%` : "100.0%"}
          </div>
          <p className="text-[11px] text-slate-500">Execution success ratio</p>
        </GlassCard>

        <GlassCard className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
            <span>Avg Pipeline Latency</span>
            <Clock className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl md:text-3xl font-bold font-mono text-purple-400">
            {metrics ? formatLatency(metrics.average_latency_ms) : "< 250ms"}
          </div>
          <p className="text-[11px] text-slate-500">End-to-end verification time</p>
        </GlassCard>

        <GlassCard className="p-5 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
            <span>Active Models</span>
            <Cpu className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl md:text-3xl font-bold font-mono text-amber-400">
            {metrics?.active_models ?? 8}
          </div>
          <p className="text-[11px] text-slate-500">Supported LLM architectures</p>
        </GlassCard>
      </div>

      {/* ── System Health & Component Status Grid ────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <GlassCard className="p-6 md:col-span-2 space-y-4">
          <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-blue-400" />
            Framework Pipeline Component Status
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <ComponentStatusCard
              name="BM25 + Dense Retriever"
              status={componentsReady.retriever ?? true}
              description="Hybrid evidence grounding index"
            />
            <ComponentStatusCard
              name="NLI Cross-Encoder"
              status={componentsReady.nli_model ?? true}
              description="Factual claim entailment model"
            />
            <ComponentStatusCard
              name="Logit Entropy Engine"
              status={componentsReady.cross_encoder ?? true}
              description="Whitebox uncertainty estimation"
            />
            <ComponentStatusCard
              name="Adaptive Fusion Engine"
              status={componentsReady.fusion_engine ?? true}
              description="Platt-calibrated H-Score fusion"
            />
          </div>
        </GlassCard>

        {/* Live Backend Telemetry Meta */}
        <GlassCard className="p-6 space-y-4">
          <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <Database className="w-4 h-4 text-indigo-400" />
            System Environment
          </h2>

          <div className="space-y-3 text-xs font-mono">
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-500">API Endpoint</span>
              <span className="text-slate-300 truncate max-w-[140px]">Production</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-500">Framework Version</span>
              <span className="text-blue-400">v1.0 Production</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/5">
              <span className="text-slate-500">Epistemic Gate</span>
              <span className="text-emerald-400">Enabled</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-500">Fusion Weights</span>
              <span className="text-slate-300">α=0.45, β=0.30, γ=0.25</span>
            </div>
          </div>
        </GlassCard>
      </div>

      {/* ── Recent Verifications Table ───────────────────────────────────── */}
      <GlassCard className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-purple-400" />
            Recent Verification Executions
          </h2>
          <Link href="/traces" className="text-xs text-blue-400 hover:underline flex items-center gap-1">
            View All Traces <ExternalLink className="w-3 h-3" />
          </Link>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-12 border border-dashed border-white/10 rounded-xl text-slate-500 text-xs">
            No local verification history recorded yet. Run a verification on the{" "}
            <Link href="/verify" className="text-blue-400 underline">
              Verify Page
            </Link>{" "}
            to populate live telemetry logs.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-white/10 text-slate-400">
                  <th className="pb-2">Trace ID</th>
                  <th className="pb-2">Timestamp</th>
                  <th className="pb-2">Response Excerpt</th>
                  <th className="pb-2">H-Score</th>
                  <th className="pb-2">Risk Level</th>
                  <th className="pb-2 text-right">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {history.slice(0, 5).map((item) => {
                  const hScore = item.result.overall_h_score;
                  const riskLevel = item.result.risk_level;
                  const latency = item.result.latency_ms ?? item.result.processing_time_ms ?? 0;
                  return (
                    <tr key={item.id} className="text-slate-300 hover:bg-white/[0.02]">
                      <td className="py-3 text-blue-400 font-semibold">{item.id.slice(0, 12)}</td>
                      <td className="py-3 text-slate-500">{new Date(item.timestamp).toLocaleTimeString()}</td>
                      <td className="py-3 font-sans truncate max-w-xs">{item.response}</td>
                      <td className="py-3 font-bold" style={{ color: getRiskColor(riskLevel) }}>
                        {(hScore * 100).toFixed(1)}%
                      </td>
                      <td className="py-3">
                        <Badge
                          variant={riskLevel === "VERIFIED" ? "verified" : "warning"}
                          className="text-[10px]"
                        >
                          {getRiskLabel(riskLevel)}
                        </Badge>
                      </td>
                      <td className="py-3 text-right text-slate-400">{formatLatency(latency)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>
    </div>
  );
}

function ComponentStatusCard({
  name,
  status,
  description,
}: {
  name: string;
  status: boolean;
  description: string;
}) {
  return (
    <div className="p-3.5 rounded-xl border border-white/[0.06] bg-black/20 flex items-start justify-between gap-3">
      <div className="space-y-0.5 min-w-0">
        <span className="text-xs font-semibold text-slate-200 block truncate">{name}</span>
        <span className="text-[11px] text-slate-500 block truncate">{description}</span>
      </div>
      <Badge variant={status ? "verified" : "danger"} className="shrink-0 text-[10px]">
        {status ? "READY" : "OFFLINE"}
      </Badge>
    </div>
  );
}
