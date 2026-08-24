"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  ShieldCheck,
  ShieldAlert,
  RefreshCw,
  Clock,
  Activity,
  CheckCircle2,
  XCircle,
  GitBranch,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Zap,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatCard } from "@/components/ui/StatCard";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SkeletonMetrics } from "@/components/ui/skeleton";
import { AwaitingTelemetry } from "@/components/ui/EmptyState";
import { useMetrics, useHealth, useReady } from "@/hooks/use-analysis";
import { useAnalysisStore } from "@/store/analysis-store";
import { formatLatency, formatNumber, formatScore, getRiskColor, getRiskLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function OverviewPage() {
  const { data: metrics, isLoading: metricsLoading, refetch: refetchMetrics } = useMetrics();
  const { data: health } = useHealth();
  const { data: ready } = useReady();
  const history = useAnalysisStore((s) => s.history);
  const router = useRouter();

  const isHealthy = health?.status === "ok" || health?.status === "healthy";
  const components = ready?.components || {};
  const hasMetrics = metrics && (metrics.requests > 0 || (metrics.total_requests && metrics.total_requests > 0));

  // Compute derived metrics from history and backend metrics
  const totalVerifications = Math.max(metrics?.requests ?? 0, history.length);
  const hallucinationsDetected = history.filter(
    (h) => h.result.risk_level === "LIKELY_HALLUCINATED" || h.result.risk_level === "MODERATE_RISK"
  ).length;
  const verificationsVerified = history.filter(
    (h) => h.result.risk_level === "VERIFIED"
  ).length;
  
  const avgHScore = (metrics && metrics.average_h_score !== null && metrics.average_h_score !== undefined)
    ? metrics.average_h_score
    : (history.length > 0
      ? history.reduce((sum, h) => sum + (h.result.overall_h_score ?? 0), 0) / history.length
      : null);

  const avgLatencyMs = (metrics && metrics.average_latency_ms !== null && metrics.average_latency_ms !== undefined)
    ? metrics.average_latency_ms
    : (history.length > 0
      ? history.reduce((sum, h) => sum + ((h.result as any).latency_ms ?? 0), 0) / history.length
      : null);

  const successRate = (metrics && metrics.success_rate !== null && metrics.success_rate !== undefined)
    ? metrics.success_rate
    : (totalVerifications > 0 ? 100.0 : null);

  // Get greeting based on time
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  // Recent issues categories
  const issueCategories = React.useMemo(() => {
    const cats: Record<string, number> = {};
    history.forEach((h) => {
      const rc = h.result.root_cause_classification;
      if (rc && rc !== "VERIFIED" && rc !== "NONE") {
        cats[rc] = (cats[rc] || 0) + 1;
      }
    });
    return Object.entries(cats)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 6);
  }, [history]);

  // Risk distribution
  const riskDist = React.useMemo(() => {
    const dist = { VERIFIED: 0, NEEDS_VERIFICATION: 0, MODERATE_RISK: 0, LIKELY_HALLUCINATED: 0 };
    history.forEach((h) => {
      const rl = h.result.risk_level;
      if (rl in dist) dist[rl as keyof typeof dist]++;
    });
    return dist;
  }, [history]);

  return (
    <div className="p-5 md:p-8 space-y-6 max-w-[1400px] mx-auto pb-20 md:pb-8">
      {/* ── Page Header ─────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <p className="text-[var(--text-muted)] text-sm mb-1">{greeting}</p>
          <h1 className="text-heading-lg text-[var(--text-primary)]">
            Command Center
          </h1>
          <p className="text-label-md text-[var(--text-muted)] mt-1">
            AI verification and hallucination intelligence
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className={cn(
            "flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius)] text-[11px] font-mono border",
            isHealthy
              ? "text-[var(--verified)] bg-[var(--verified-soft)] border-[var(--verified-border)]"
              : "text-[var(--hallucination)] bg-[var(--hallucination-soft)] border-[var(--hallucination-border)]"
          )}>
            <span className={cn("w-1.5 h-1.5 rounded-full", isHealthy ? "bg-[var(--verified)] animate-pulse-dot" : "bg-[var(--hallucination)]")} />
            {isHealthy ? "All systems operational" : "Systems offline"}
          </div>
          <Button variant="ghost" size="icon-sm" onClick={() => refetchMetrics()} aria-label="Refresh metrics">
            <RefreshCw className={cn("w-3.5 h-3.5", metricsLoading && "animate-spin")} />
          </Button>
        </div>
      </div>

      {/* ── KPI Cards ───────────────────────────────────────────────── */}
      {metricsLoading ? (
        <SkeletonMetrics count={6} />
      ) : (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--text-dim)]">
              Live Production Telemetry
            </span>
            <span className="h-px flex-1 bg-[var(--border)]" />
          </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <StatCard
            label="Verifications"
            value={totalVerifications > 0 ? formatNumber(totalVerifications) : "—"}
            caption={totalVerifications > 0 ? "Total analyses run" : "Awaiting first verification"}
            icon={ShieldCheck}
            status="default"
          />
          <StatCard
            label="Hallucinations"
            value={totalVerifications > 0 ? formatNumber(hallucinationsDetected) : "—"}
            caption={totalVerifications > 0 ? `${hallucinationsDetected} detected across analyses` : "Awaiting data"}
            icon={XCircle}
            status={hallucinationsDetected > 0 ? "hallucination" : "default"}
          />
          <StatCard
            label="Verified"
            value={totalVerifications > 0 ? formatNumber(verificationsVerified) : "—"}
            caption={totalVerifications > 0 ? `${verificationsVerified} claims verified safe` : "Awaiting data"}
            icon={CheckCircle2}
            status={verificationsVerified > 0 ? "verified" : "default"}
          />
          <StatCard
            label="Success Rate"
            value={successRate !== null ? `${successRate.toFixed(1)}%` : "—"}
            caption={totalVerifications > 0 ? "Pipeline execution rate" : "Awaiting telemetry"}
            icon={Activity}
            status="default"
          />
          <StatCard
            label="Avg H-Score"
            value={avgHScore !== null ? formatScore(avgHScore) + "%" : "—"}
            caption={avgHScore !== null ? "Mean hallucination score" : "Awaiting data"}
            icon={BarChart3}
            status={avgHScore !== null ? (avgHScore > 0.5 ? "hallucination" : avgHScore > 0.25 ? "warning" : "verified") : "default"}
          />
          <StatCard
            label="Avg Latency"
            value={avgLatencyMs !== null && avgLatencyMs > 0 ? formatLatency(avgLatencyMs) : "—"}
            caption={avgLatencyMs !== null && avgLatencyMs > 0 ? "Pipeline execution time" : "Awaiting telemetry"}
            icon={Clock}
            status="default"
          />
        </div>
        </div>
      )}

      {/* ── Component Health + Risk Distribution ─────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Component Health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-[var(--text-muted)]" />
              Pipeline Components
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {Object.entries(components).length > 0 ? (
              Object.entries(components).map(([name, ready]) => (
                <div key={name} className="flex items-center justify-between py-1.5 px-1">
                  <span className="text-[13px] text-[var(--text-secondary)] font-mono">
                    {name.replace(/_/g, " ")}
                  </span>
                  <StatusBadge status={ready ? "verified" : "failed"} size="sm" />
                </div>
              ))
            ) : (
              <div className="text-sm text-[var(--text-muted)] py-4 text-center">
                {isHealthy ? "Components operational" : "Awaiting readiness data"}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Risk Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-[var(--text-muted)]" />
              Verification Outcomes
            </CardTitle>
          </CardHeader>
          <CardContent>
            {history.length > 0 ? (
              <div className="space-y-3">
                {Object.entries(riskDist).map(([level, count]) => {
                  const total = history.length;
                  const pct = total > 0 ? (count / total) * 100 : 0;
                  return (
                    <div key={level} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-[var(--text-secondary)]">{getRiskLabel(level)}</span>
                        <span className="font-mono text-[var(--text-muted)]">{count} ({pct.toFixed(0)}%)</span>
                      </div>
                      <div className="h-1.5 bg-[var(--surface)] rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${pct}%` }}
                          transition={{ duration: 0.6, ease: "easeOut" }}
                          className="h-full rounded-full"
                          style={{ backgroundColor: getRiskColor(level) }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <AwaitingTelemetry />
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Recent Issues + Recent Activity ──────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Recent Issues */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-[var(--text-muted)]" />
                Recent Issues
              </CardTitle>
              {issueCategories.length > 0 && (
                <Link href="/errors">
                  <Button variant="ghost" size="sm" className="text-xs">
                    View all <ArrowRight className="w-3 h-3" />
                  </Button>
                </Link>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {issueCategories.length > 0 ? (
              <div className="space-y-2">
                {issueCategories.map(([category, count]) => (
                  <div
                    key={category}
                    className="flex items-center justify-between py-2 px-2 rounded-[var(--radius)] hover:bg-[var(--surface-hover)] transition-colors"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="w-1.5 h-1.5 rounded-full bg-[var(--hallucination)] shrink-0" />
                      <span className="text-[13px] text-[var(--text-secondary)] truncate">
                        {formatIssueCategory(category)}
                      </span>
                    </div>
                    <Badge variant="outline" size="sm">{count}</Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-6 text-center">
                <ShieldCheck className="w-8 h-8 text-[var(--text-dim)] mx-auto mb-2" />
                <p className="text-sm text-[var(--text-muted)]">No issues detected</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-[var(--text-muted)]" />
                Recent Activity
              </CardTitle>
              {history.length > 0 && (
                <Link href="/traces">
                  <Button variant="ghost" size="sm" className="text-xs">
                    View traces <ArrowRight className="w-3 h-3" />
                  </Button>
                </Link>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {history.length > 0 ? (
              <div className="space-y-1">
                {history.slice(0, 6).map((entry, i) => (
                  <button
                    key={entry.id}
                    onClick={() => {
                      useAnalysisStore.getState().setSelectedTraceId(entry.id);
                      router.push("/traces");
                    }}
                    className="w-full flex items-center gap-3 py-2 px-2 rounded-[var(--radius)] hover:bg-[var(--surface-hover)] transition-colors text-left cursor-pointer"
                  >
                    <StatusBadge status={entry.result.risk_level} size="sm" showIcon={true} />
                    <span className="text-[13px] text-[var(--text-secondary)] truncate flex-1 min-w-0">
                      {truncateText(entry.query || entry.response, 50)}
                    </span>
                    <span className="text-[11px] font-mono text-[var(--text-dim)] shrink-0">
                      {formatScore(entry.result.overall_h_score)}%
                    </span>
                  </button>
                ))}
              </div>
            ) : (
              <AwaitingTelemetry />
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Quick Actions ────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <Link href="/verify">
          <Card variant="interactive" className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-[var(--radius)] bg-[var(--ai-soft)] border border-[var(--ai-border)] flex items-center justify-center">
                <ShieldCheck className="w-4 h-4 text-[var(--ai)]" />
              </div>
              <div>
                <p className="text-sm font-semibold text-[var(--text-primary)]">Verify a claim</p>
                <p className="text-[11px] text-[var(--text-muted)]">Run verification pipeline</p>
              </div>
            </div>
          </Card>
        </Link>
        <Link href="/traces">
          <Card variant="interactive" className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-[var(--radius)] bg-[var(--evidence-soft)] border border-[var(--evidence-border)] flex items-center justify-center">
                <GitBranch className="w-4 h-4 text-[var(--evidence)]" />
              </div>
              <div>
                <p className="text-sm font-semibold text-[var(--text-primary)]">View traces</p>
                <p className="text-[11px] text-[var(--text-muted)]">Inspect pipeline execution</p>
              </div>
            </div>
          </Card>
        </Link>
        <Link href="/scientific">
          <Card variant="interactive" className="p-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-[var(--radius)] bg-[var(--verified-soft)] border border-[var(--verified-border)] flex items-center justify-center">
                <BarChart3 className="w-4 h-4 text-[var(--verified)]" />
              </div>
              <div>
                <p className="text-sm font-semibold text-[var(--text-primary)]">Scientific Lab</p>
                <p className="text-[11px] text-[var(--text-muted)]">Research benchmark results</p>
              </div>
            </div>
          </Card>
        </Link>
      </div>
    </div>
  );
}

function truncateText(text: string, maxLen: number = 60) {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "…";
}

function formatIssueCategory(category: string): string {
  return category
    .replace(/_/g, " ")
    .replace(/([A-Z])/g, " $1")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
