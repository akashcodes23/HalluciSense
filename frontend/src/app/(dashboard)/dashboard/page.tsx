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
import { Card } from "@/components/ui/card";
import { StatCard } from "@/components/ui/StatCard";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { useMetrics, useHealth, useReady } from "@/hooks/use-analysis";
import { useAnalysisStore } from "@/store/analysis-store";
import { formatLatency, formatNumber, getRiskColor, getRiskLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { useRouter } from "next/navigation";

const truncateText = (text: string, maxLen: number = 60) => {
  if (text.length <= maxLen) return text;
  const truncated = text.slice(0, maxLen);
  const lastSpace = truncated.lastIndexOf(" ");
  if (lastSpace > maxLen * 0.75) {
    return truncated.slice(0, lastSpace) + "...";
  }
  return truncated + "...";
};

export default function DashboardPage() {
  const { data: metrics, isLoading: metricsLoading, refetch: refetchMetrics } = useMetrics();
  const { data: health } = useHealth();
  const { data: ready } = useReady();
  const history = useAnalysisStore((s) => s.history);
  const setSelectedTraceId = useAnalysisStore((s) => s.setSelectedTraceId);
  const router = useRouter();

  const isHealthy = health?.status === "ok" || health?.status === "healthy";
  const componentsReady = ready?.components || {};

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-7xl mx-auto">
      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.04] pb-6">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/[0.04] bg-white/[0.02] text-slate-400 text-xs font-semibold tracking-wide uppercase mb-2 font-mono">
            <LayoutDashboard className="w-3.5 h-3.5" />
            System Telemetry Overview
          </div>
          <h1 className="text-heading-md md:text-heading-lg font-bold text-white tracking-tight leading-none">
            HalluciSense Control Center
          </h1>
          <p className="text-label-md text-slate-500">
            Live execution statistics, framework component health, historical verification logs, and active risk distribution.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-white/[0.04] bg-bg-surface text-xs font-mono">
            <span className={cn("w-2 h-2 rounded-full", isHealthy ? "bg-status-success animate-pulse" : "bg-status-error")} />
            <span className="text-slate-400">
              Backend: {isHealthy ? "ONLINE" : "DISCONNECTED"}
            </span>
          </div>

          <button
            onClick={() => refetchMetrics()}
            className="p-2 rounded-xl border border-white/5 bg-white/[0.01] text-slate-500 hover:text-white transition-colors cursor-pointer"
            title="Refresh Telemetry"
          >
            <RefreshCw className={`w-4 h-4 ${metricsLoading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* ── Key Metrics Cards Row ───────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Total Requests"
          value={formatNumber(Math.max(metrics?.requests ?? 0, history.length))}
          caption={metrics && metrics.requests > 0 ? "Server-side total" : history.length > 0 ? "From local session history" : "No requests recorded"}
          icon={Activity}
          status="default"
        />

        <StatCard
          label="Success Rate"
          value={metrics && metrics.requests > 0 && metrics.success_rate !== null ? `${metrics.success_rate.toFixed(1)}%` : "—"}
          caption={metrics && metrics.requests > 0 ? "Execution success ratio" : "Awaiting first execution"}
          icon={CheckCircle2}
          status={metrics && metrics.success_rate !== null && metrics.success_rate < 95 ? "warning" : "default"}
        />

        <StatCard
          label="Avg Pipeline Latency"
          value={metrics && metrics.requests > 0 && metrics.average_latency_ms !== null ? formatLatency(metrics.average_latency_ms) : "—"}
          caption={metrics && metrics.requests > 0 ? "End-to-end verification time" : "Awaiting first execution"}
          icon={Clock}
          status="default"
        />

        <StatCard
          label="Session Traces"
          value={formatNumber(history.length)}
          caption="Local verification executions"
          icon={GitBranch}
          status="default"
        />
      </div>

      {/* ── System Health & Component Status Grid ────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="p-6 md:col-span-2 space-y-4">
          <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-slate-400" />
            Framework Pipeline Component Status
          </h2>

          <div className="divide-y divide-white/[0.04] space-y-1">
            <ComponentStatusRow
              name="BM25 + Dense Retriever"
              status={componentsReady.retriever ?? true}
              description="Hybrid evidence grounding index"
            />
            <ComponentStatusRow
              name="NLI Cross-Encoder"
              status={componentsReady.nli_model ?? true}
              description="Factual claim entailment model"
            />
            <ComponentStatusRow
              name="Logit Entropy Engine"
              status={componentsReady.cross_encoder ?? true}
              description="Whitebox uncertainty estimation"
            />
            <ComponentStatusRow
              name="Adaptive Fusion Engine"
              status={componentsReady.fusion_engine ?? true}
              description="Platt-calibrated H-Score fusion"
            />
          </div>
        </Card>

        {/* Live Backend Telemetry Meta */}
        <Card className="p-6 space-y-4">
          <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <Database className="w-4 h-4 text-slate-400" />
            System Environment
          </h2>

          <div className="space-y-3 text-xs font-mono">
            <div className="flex justify-between py-1.5 border-b border-white/[0.04]">
              <span className="text-slate-500">API Endpoint</span>
              <span className="text-slate-300 truncate max-w-[140px]">Production</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/[0.04]">
              <span className="text-slate-500">Framework Version</span>
              <span className="text-accent-primary font-semibold">v1.0 Production</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-white/[0.04]">
              <span className="text-slate-500">Epistemic Gate</span>
              <span className="text-status-success font-semibold">Enabled</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-500">Fusion Weights</span>
              <span className="text-slate-300">α=0.45, β=0.30, γ=0.25</span>
            </div>
          </div>
        </Card>
      </div>

      {/* ── Recent Verifications Table ───────────────────────────────────── */}
      <Card className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-slate-400" />
            Recent Verification Executions
          </h2>
          <Link href="/traces" className="text-xs text-accent-primary hover:underline flex items-center gap-1">
            View All Traces <ExternalLink className="w-3 h-3" />
          </Link>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-12 border border-dashed border-white/10 rounded-xl text-slate-500 text-xs">
            No local verification history recorded yet. Run a verification on the{" "}
            <Link href="/verify" className="text-accent-primary underline">
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
                      <td className="py-3 font-semibold">
                        <button
                          onClick={() => {
                            setSelectedTraceId(item.id);
                            router.push("/traces");
                          }}
                          className="text-accent-primary hover:underline cursor-pointer text-left font-mono"
                        >
                          {item.id.slice(0, 12)}
                        </button>
                      </td>
                      <td className="py-3 text-slate-500">{new Date(item.timestamp).toLocaleTimeString()}</td>
                      <td className="py-3 font-sans truncate max-w-xs" title={item.response}>
                        {truncateText(item.response)}
                      </td>
                      <td className="py-3 font-bold font-mono" style={{ color: getRiskColor(riskLevel) }}>
                        {(hScore * 100).toFixed(1)}%
                      </td>
                      <td className="py-3">
                        <StatusBadge
                          label={getRiskLabel(riskLevel)}
                          status={
                            riskLevel === "VERIFIED"
                              ? "success"
                              : riskLevel === "LIKELY_HALLUCINATED"
                              ? "error"
                              : "warning"
                          }
                        />
                      </td>
                      <td className="py-3 text-right text-slate-400">{formatLatency(latency)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

function ComponentStatusRow({
  name,
  status,
  description,
}: {
  name: string;
  status: boolean;
  description: string;
}) {
  return (
    <div className="flex items-center justify-between py-3 gap-3">
      <div className="space-y-0.5 min-w-0">
        <span className="text-xs font-semibold text-slate-200 block truncate">{name}</span>
        <span className="text-[11px] text-slate-500 block truncate">{description}</span>
      </div>
      <StatusBadge
        label={status ? "READY" : "OFFLINE"}
        status={status ? "success" : "error"}
        className="shrink-0"
      />
    </div>
  );
}
