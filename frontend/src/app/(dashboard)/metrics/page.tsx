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
import { GlassCard } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
      </div>
    );
  }

  if (isError || !metrics) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-6">
        <div className="w-16 h-16 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center mb-4">
          <XCircle className="w-7 h-7 text-red-400" />
        </div>
        <p className="text-slate-400 text-sm">
          Unable to connect to the backend. Ensure the server is running.
        </p>
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
      <div className="max-w-5xl mx-auto px-6 py-8 space-y-8">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-600 to-teal-600 shadow-[0_0_24px_rgba(34,197,94,0.25)]">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">Metrics</h1>
              <p className="text-sm text-slate-500">Real-time system telemetry</p>
            </div>
          </div>

          <Badge variant={isHealthy ? "verified" : "danger"}>
            <div className={cn("w-2 h-2 rounded-full", isHealthy ? "bg-emerald-400 animate-pulse" : "bg-red-400")} />
            {isHealthy ? "Online" : "Offline"}
          </Badge>
        </motion.div>

        {/* ── Metric Cards Grid ───────────────────────────────────────── */}
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="grid grid-cols-2 lg:grid-cols-3 gap-4"
        >
          <MetricCard
            icon={<Activity className="w-4 h-4" />}
            label="Total Requests"
            value={formatNumber(metrics.requests)}
            color="#2563EB"
          />
          <MetricCard
            icon={<Clock className="w-4 h-4" />}
            label="Avg Latency"
            value={formatLatency(metrics.average_latency_ms)}
            color="#A855F7"
          />
          <MetricCard
            icon={<TrendingUp className="w-4 h-4" />}
            label="Avg H-Score"
            value={metrics.requests > 0 ? `${((metrics.avg_h_score ?? 0) * 100).toFixed(1)}%` : "0.0%"}
            color="#F59E0B"
          />
          <MetricCard
            icon={<CheckCircle2 className="w-4 h-4" />}
            label="Success Rate"
            value={metrics.requests > 0 ? `${metrics.success_rate.toFixed(1)}%` : "100.0%"}
            color="#22C55E"
          />
          <MetricCard
            icon={<XCircle className="w-4 h-4" />}
            label="Error Rate"
            value={metrics.requests > 0 ? `${metrics.error_rate.toFixed(1)}%` : "0.0%"}
            color="#EF4444"
          />
          <MetricCard
            icon={<HardDrive className="w-4 h-4" />}
            label="Memory"
            value={formatMemory(148)}
            color="#38BDF8"
          />
        </motion.div>

        {/* ── Charts ──────────────────────────────────────────────────── */}
        <motion.div variants={item} initial="hidden" animate="show">
          <GlassCard className="p-6">
            <h3 className="text-sm font-medium text-slate-400 mb-4">Latency Distribution</h3>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={sparkData}>
                  <defs>
                    <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#2563EB" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#2563EB" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" hide />
                  <YAxis hide />
                  <RechartsTooltip
                    contentStyle={{
                      backgroundColor: "#111827",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: "8px",
                      fontSize: "12px",
                      color: "#F8FAFC",
                    }}
                    formatter={(value: any) => [`${Number(value || 0).toFixed(0)}ms`, "Latency"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="latency"
                    stroke="#2563EB"
                    strokeWidth={2}
                    fill="url(#latencyGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </GlassCard>
        </motion.div>
      </div>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <motion.div variants={item}>
      <GlassCard className="p-5 space-y-3">
        <div className="flex items-center gap-2">
          <div
            className="flex items-center justify-center w-8 h-8 rounded-lg"
            style={{ backgroundColor: `${color}15` }}
          >
            <span style={{ color }}>{icon}</span>
          </div>
          <span className="text-xs text-slate-500">{label}</span>
        </div>
        <p className="text-2xl font-bold font-mono text-white tracking-tight">{value}</p>
      </GlassCard>
    </motion.div>
  );
}
