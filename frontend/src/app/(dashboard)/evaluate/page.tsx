"use client";

import React, { useState } from "react";
import {
  BarChart3,
  Database,
  GitCompare,
  Play,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/EmptyState";
import { useMetrics } from "@/hooks/use-analysis";
import { cn } from "@/lib/utils";
import Link from "next/link";

const TABS = [
  { id: "runs", label: "Evaluation Runs", icon: Play },
  { id: "datasets", label: "Datasets", icon: Database },
  { id: "benchmarks", label: "Benchmarks", icon: BarChart3 },
  { id: "comparisons", label: "Comparisons", icon: GitCompare },
];

// Frozen research results from Phase 10 — clearly labeled
const RESEARCH_BENCHMARKS = [
  {
    name: "Phase 10 — Independent Generalization",
    dataset: "HalluciSense Canonical Benchmark",
    model: "Hybrid Fusion (P1 + Symbolic)",
    auroc: "0.9855",
    auprc: "—",
    f1: "0.9479",
    accuracy: "94.80%",
    ece: "0.0520",
    status: "completed",
    created: "2026-08-05",
  },
];

export default function EvaluatePage() {
  const [activeTab, setActiveTab] = useState("benchmarks");
  const { data: metrics } = useMetrics();

  return (
    <div className="p-5 md:p-8 space-y-6 max-w-[1200px] mx-auto pb-20 md:pb-8">
      {/* Header */}
      <div>
        <h1 className="text-heading-lg text-[var(--text-primary)]">Evaluate</h1>
        <p className="text-label-md text-[var(--text-muted)] mt-1">
          Evaluation workspace for verification model performance
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-1 border-b border-[var(--border)]">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "flex items-center gap-2 px-4 py-2.5 text-[13px] font-medium transition-colors cursor-pointer",
                "border-b-2 -mb-px",
                activeTab === tab.id
                  ? "border-[var(--primary)] text-[var(--primary)]"
                  : "border-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              )}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      {activeTab === "runs" && (
        <EmptyState
          icon={Play}
          title="No evaluation runs recorded"
          description="Evaluation runs will appear here when the evaluation API is available. Use the Scientific Lab to view frozen research benchmark results."
          action={{ label: "View Scientific Lab", onClick: () => window.location.href = "/scientific" }}
        />
      )}

      {activeTab === "datasets" && (
        <Card>
          <CardHeader>
            <CardTitle>Available Datasets</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-[var(--radius)] border border-[var(--border)] overflow-hidden">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Dataset</th>
                    <th>Samples</th>
                    <th>Domains</th>
                    <th>Categories</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="text-[var(--text-primary)] font-medium">HalluciSense Canonical Benchmark</td>
                    <td className="font-mono">200</td>
                    <td>5</td>
                    <td>8</td>
                    <td><Badge variant="verified">Frozen</Badge></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-[11px] text-[var(--text-dim)] mt-3">
              SHA-256: dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5
            </p>
          </CardContent>
        </Card>
      )}

      {activeTab === "benchmarks" && (
        <div className="space-y-4">
          <Badge variant="warning" size="lg">
            Research Benchmark Results — Not Live Production Telemetry
          </Badge>
          <Card>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Benchmark</th>
                      <th>Dataset</th>
                      <th>Model</th>
                      <th>AUROC</th>
                      <th>F1</th>
                      <th>Accuracy</th>
                      <th>ECE</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {RESEARCH_BENCHMARKS.map((b, i) => (
                      <tr key={i}>
                        <td className="text-[var(--text-primary)] font-medium">{b.name}</td>
                        <td>{b.dataset}</td>
                        <td className="font-mono text-[12px]">{b.model}</td>
                        <td className="font-mono font-bold text-[var(--primary)]">{b.auroc}</td>
                        <td className="font-mono">{b.f1}</td>
                        <td className="font-mono">{b.accuracy}</td>
                        <td className="font-mono">{b.ece}</td>
                        <td><Badge variant="verified">{b.status}</Badge></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Frozen Validation Metrics from backend */}
          {metrics && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">System Validation Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <MetricCell label="Total Requests" value={metrics.requests?.toString() ?? "—"} />
                  <MetricCell label="Success Rate" value={metrics.success_rate != null ? `${metrics.success_rate.toFixed(1)}%` : "—"} />
                  <MetricCell label="Avg H-Score" value={metrics.avg_h_score != null ? `${(metrics.avg_h_score * 100).toFixed(1)}%` : "—"} />
                  <MetricCell label="Avg Latency" value={metrics.average_latency_ms != null ? `${metrics.average_latency_ms.toFixed(0)}ms` : "—"} />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {activeTab === "comparisons" && (
        <EmptyState
          icon={GitCompare}
          title="No comparisons available"
          description="Model comparisons will be available when multiple evaluation runs have been recorded."
        />
      )}
    </div>
  );
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[var(--radius)] bg-[var(--surface)] p-3 text-center">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">{label}</p>
      <p className="text-sm font-mono font-medium text-[var(--text-primary)]">{value}</p>
    </div>
  );
}
