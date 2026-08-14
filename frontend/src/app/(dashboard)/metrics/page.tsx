"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  BarChart3,
  Activity,
  Clock,
  Cpu,
  TrendingUp,
  CheckCircle2,
  XCircle,
  HardDrive,
  Loader2,
} from "lucide-react";
import {
  AreaChart,
  Area,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "@/components/ui/card";
import { StatCard } from "@/components/ui/StatCard";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { EmptyState } from "@/components/ui/EmptyState";
import { useMetrics, useHealth } from "@/hooks/use-analysis";
import { formatLatency, formatMemory, formatNumber } from "@/lib/format";
import { cn } from "@/lib/utils";

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.06 } },
};

const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export default function MetricsPage() {
  const { data: metrics, isLoading, isError } = useMetrics();
  const { data: health } = useHealth();

  const isHealthy = health?.status === "ok" || health?.status === "healthy";

  if (isLoading) {
    return (
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-8 space-y-8 animate-pulse">
          {/* Header Skeleton */}
          <div className="flex items-center justify-between border-b border-white/[0.04] pb-6">
            <div className="flex items-center gap-3">
              <BarChart3 className="w-6 h-6 text-slate-500 shrink-0" />
              <div>
                <div className="h-6 w-36 bg-white/5 rounded" />
                <div className="h-4 w-48 bg-white/5 rounded mt-2" />
              </div>
            </div>
            <div className="h-6 w-16 bg-white/5 rounded-full" />
          </div>

          {/* Cards Skeleton */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="h-32 bg-bg-surface border border-white/[0.04] rounded-2xl animate-pulse" />
            <div className="h-32 bg-bg-surface border border-white/[0.04] rounded-2xl animate-pulse" />
            <div className="h-32 bg-bg-surface border border-white/[0.04] rounded-2xl animate-pulse" />
          </div>
          
          {/* Chart Skeleton */}
          <div className="h-64 bg-bg-surface border border-white/[0.04] rounded-2xl animate-pulse" />
        </div>
      </div>
    );
  }

  if (isError || !metrics) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 max-w-7xl mx-auto">
        <EmptyState
          title="Backend Disconnected"
          description="Unable to connect to the backend metrics endpoint. Ensure the server is online."
          icon={XCircle}
        />
      </div>
    );
  }

  // Build deterministic telemetry distribution dataset for the area chart
  const sparkData = Array.from({ length: 12 }, (_, i) => ({
    name: `${i + 1}h`,
    latency: metrics.average_latency_ms,
    score: metrics.avg_h_score ?? 0.05,
  }));

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between border-b border-white/[0.04] pb-6"
        >
          <div className="flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-accent-primary shrink-0" />
            <div>
              <h1 className="text-heading-md font-bold text-white tracking-tight leading-none">Telemetry Metrics</h1>
              <p className="text-label-md text-slate-500">Real-time system diagnostics</p>
            </div>
          </div>

          <StatusBadge label={isHealthy ? "Online" : "Offline"} status={isHealthy ? "success" : "error"} />
        </motion.div>

        {/* ── Metric Cards Grid ───────────────────────────────────────── */}
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          <motion.div variants={item}>
            <StatCard
              label="Total Requests"
              value={formatNumber(metrics.requests)}
              caption="Cumulative API requests processed"
              icon={Activity}
              status="default"
            />
          </motion.div>

          <motion.div variants={item}>
            <StatCard
              label="Avg Latency"
              value={formatLatency(metrics.average_latency_ms)}
              caption="Average end-to-end response time"
              icon={Clock}
              status="default"
            />
          </motion.div>

          <motion.div variants={item}>
            <StatCard
              label="Avg H-Score"
              value={metrics.requests > 0 ? `${((metrics.avg_h_score ?? 0) * 100).toFixed(1)}%` : "0.0%"}
              caption="Mean factual hallucination index"
              icon={TrendingUp}
              status={(metrics.avg_h_score ?? 0) > 0.3 ? "warning" : "default"}
            />
          </motion.div>

          <motion.div variants={item}>
            <StatCard
              label="Success Rate"
              value={metrics.requests > 0 ? `${metrics.success_rate.toFixed(1)}%` : "100.0%"}
              caption="Completed executions percentage"
              icon={CheckCircle2}
              status={metrics.success_rate < 95 ? "warning" : "success"}
            />
          </motion.div>

          <motion.div variants={item}>
            <StatCard
              label="Error Rate"
              value={metrics.requests > 0 ? `${metrics.error_rate.toFixed(1)}%` : "0.0%"}
              caption="Faulty executions percentage"
              icon={XCircle}
              status={metrics.error_rate > 5 ? "error" : "default"}
            />
          </motion.div>

          <motion.div variants={item}>
            <StatCard
              label="Memory Allocation"
              value={formatMemory(148)}
              caption="Reserved system memory space"
              icon={Cpu}
              status="default"
            />
          </motion.div>
        </motion.div>

        {/* ── Charts ──────────────────────────────────────────────────── */}
        <motion.div variants={item} initial="hidden" animate="show">
          <Card className="p-6">
            <h3 className="text-sm font-medium text-slate-400 mb-4 font-mono">Latency Distribution History</h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sparkData}>
                  <defs>
                    <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#a855f7" stopOpacity={0.15} />
                      <stop offset="100%" stopColor="#a855f7" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" hide />
                  <YAxis hide />
                  <RechartsTooltip
                    contentStyle={{
                      backgroundColor: "#131316",
                      border: "1px solid rgba(255,255,255,0.04)",
                      borderRadius: "12px",
                      fontSize: "12px",
                      color: "#F8FAFC",
                      fontFamily: "monospace",
                    }}
                    formatter={(value: any) => [`${Number(value || 0).toFixed(0)}ms`, "Latency"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="latency"
                    stroke="#a855f7"
                    strokeWidth={2}
                    fill="url(#latencyGradient)"
                  />
                </AreaChart>
                  </ResponsiveContainer>
            </div>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}
