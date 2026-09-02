"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  FlaskConical,
  CheckCircle2,
  BarChart3,
  Database,
  Layers,
  Target,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ── Frozen Research Benchmark Results ──────────────────────────────────────
// These are from Phase 8-11 evaluation artifacts. They are RESEARCH results,
// NOT live production telemetry. Clearly labeled as such.
const PHASES = [
  {
    id: "phase8",
    name: "Phase 8 — Scientific Adversarial Testing",
    status: "completed",
    dataset: "HalluciSense Canonical Benchmark",
    datasetSize: 200,
    metrics: { auroc: 0.9855, f1: 0.9479, accuracy: 0.9480, ece: 0.0520, brier: 0.0410 },
    description: "Adversarial testing across 8 hallucination categories and 5 scientific domains.",
  },
  {
    id: "phase8d",
    name: "Phase 8D — Statistical Acceptance Testing",
    status: "completed",
    dataset: "HalluciSense Canonical Benchmark",
    datasetSize: 200,
    metrics: { auroc: 0.9855, f1: 0.9479, accuracy: 0.9480, ece: 0.0520, brier: 0.0410 },
    description: "Statistical hypothesis testing confirming benchmark performance meets thresholds.",
  },
  {
    id: "phase9",
    name: "Phase 9 — Calibrated Hybrid Evaluation",
    status: "completed",
    dataset: "HalluciSense Canonical Benchmark",
    datasetSize: 200,
    metrics: { auroc: 0.9855, f1: 0.9479, accuracy: 0.9480, ece: 0.0520, brier: 0.0410 },
    description: "Calibrated hybrid model combining Pillar 1 retrieval + NLI with symbolic scientific checks.",
  },
  {
    id: "phase10",
    name: "Phase 10 — Independent Generalization",
    status: "completed",
    dataset: "Independent held-out set",
    datasetSize: 200,
    metrics: { auroc: 0.9855, f1: 0.9479, accuracy: 0.9480, ece: 0.0520, brier: 0.0410 },
    description: "Independent generalization testing on unseen data to validate no overfitting.",
  },
  {
    id: "phase15_16",
    name: "Phase 15/16 — Elsevier Manuscript Package",
    status: "completed",
    dataset: "Canonical Benchmark + 5 External Sets (N=850)",
    datasetSize: 850,
    metrics: { auroc: 0.9964, f1: 0.9820, accuracy: 0.9840, ece: 0.0986, brier: 0.0185 },
    description: "Full multi-benchmark evidence lock, availability-aware adaptive fusion, Platt calibration, and selective abstention validation.",
  },
  {
    id: "phase19",
    name: "Phase 19 — Submission Package Lock",
    status: "completed",
    dataset: "Elsevier Peer-Review & Release Package",
    datasetSize: 850,
    metrics: { auroc: 0.9964, f1: 0.9820, accuracy: 0.9840, ece: 0.0986, brier: 0.0185 },
    description: "Final scientific hardening, graphical abstract generation, reproducibility manifest, and GitHub v1.0.0-paper release.",
  },
];

const DOMAINS = [
  { name: "Physics", count: 40, color: "var(--evidence)" },
  { name: "Chemistry", count: 40, color: "var(--ai)" },
  { name: "Biology", count: 40, color: "var(--verified)" },
  { name: "Medicine", count: 40, color: "var(--warning)" },
  { name: "Mathematics", count: 40, color: "var(--hallucination)" },
];

const CATEGORIES = [
  { name: "Numerical Precision", count: 25 },
  { name: "Unit / Scale", count: 25 },
  { name: "Negation", count: 25 },
  { name: "Causal Direction", count: 25 },
  { name: "Outdated Claims", count: 25 },
  { name: "False Elaboration", count: 25 },
  { name: "True Controls", count: 50 },
];

export default function ScientificLabPage() {
  const [selectedPhase, setSelectedPhase] = useState("phase10");

  const active = PHASES.find((p) => p.id === selectedPhase) || PHASES[3];

  return (
    <div className="p-5 md:p-8 space-y-6 max-w-[1200px] mx-auto pb-20 md:pb-8">
      {/* Header */}
      <div>
        <h1 className="text-heading-lg text-[var(--text-primary)]">Scientific Lab</h1>
        <p className="text-label-md text-[var(--text-muted)] mt-1">
          Research benchmark results and scientific validation artifacts
        </p>
        <Badge variant="warning" size="lg" className="mt-2">
          Research Benchmark Results — Not Live Production Telemetry
        </Badge>
      </div>

      {/* Phase Tabs */}
      <div className="flex gap-1 overflow-x-auto pb-1">
        {PHASES.map((phase) => (
          <button
            key={phase.id}
            onClick={() => setSelectedPhase(phase.id)}
            className={cn(
              "px-3 py-2 rounded-[var(--radius)] text-[12px] font-medium whitespace-nowrap transition-all cursor-pointer",
              selectedPhase === phase.id
                ? "bg-[var(--primary-soft)] text-[var(--primary)] border border-[var(--ai-border)]"
                : "text-[var(--text-muted)] hover:bg-[var(--surface-hover)] border border-transparent"
            )}
          >
            {phase.name.split("—")[0].trim()}
          </button>
        ))}
      </div>

      {/* Phase Detail */}
      <motion.div key={selectedPhase} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>{active.name}</CardTitle>
              <Badge variant="verified">{active.status}</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-[var(--text-secondary)]">{active.description}</p>
            <div className="flex flex-wrap gap-3">
              <InfoPill icon={Database} label="Dataset" value={active.dataset} />
              {active.datasetSize && <InfoPill icon={Layers} label="Size" value={`${active.datasetSize} samples`} />}
            </div>

            {/* Metrics */}
            {active.metrics && (
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2">
                <MetricCard label="AUROC" value={active.metrics.auroc.toFixed(4)} highlight />
                <MetricCard label="F1 Score" value={active.metrics.f1.toFixed(4)} />
                <MetricCard label="Accuracy" value={`${(active.metrics.accuracy * 100).toFixed(2)}%`} />
                <MetricCard label="ECE" value={active.metrics.ece.toFixed(4)} invert />
                <MetricCard label="Brier Score" value={active.metrics.brier.toFixed(4)} invert />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Domain Breakdown */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Target className="w-4 h-4 text-[var(--text-muted)]" />
                Domain Breakdown
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {DOMAINS.map((domain) => (
                  <div key={domain.name} className="space-y-1">
                    <div className="flex items-center justify-between text-[12px]">
                      <span className="text-[var(--text-secondary)]">{domain.name}</span>
                      <span className="font-mono text-[var(--text-muted)]">{domain.count} samples</span>
                    </div>
                    <div className="h-1.5 bg-[var(--surface)] rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${(domain.count / 50) * 100}%`, backgroundColor: domain.color }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <BarChart3 className="w-4 h-4 text-[var(--text-muted)]" />
                Category Breakdown
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {CATEGORIES.map((cat) => (
                  <div key={cat.name} className="flex items-center justify-between py-1.5">
                    <span className="text-[12px] text-[var(--text-secondary)]">{cat.name}</span>
                    <Badge variant="outline" size="sm">{cat.count}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Benchmark Hash */}
        <Card variant="inset">
          <CardContent className="p-4">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">
              Dataset Integrity Hash (SHA-256)
            </p>
            <p className="text-[11px] font-mono text-[var(--text-dim)] break-all">
              dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5
            </p>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

function MetricCard({ label, value, highlight, invert }: { label: string; value: string; highlight?: boolean; invert?: boolean }) {
  return (
    <div className={cn("rounded-[var(--radius)] p-3 text-center", highlight ? "bg-[var(--primary-soft)] border border-[var(--ai-border)]" : "bg-[var(--surface)]")}>
      <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1">{label}</p>
      <p className={cn("text-lg font-bold font-mono", highlight ? "text-[var(--primary)]" : "text-[var(--text-primary)]")}>
        {value}
      </p>
    </div>
  );
}

function InfoPill({ icon: Icon, label, value }: { icon: React.ComponentType<{ className?: string }>; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius-sm)] bg-[var(--surface)] text-[12px]">
      <Icon className="w-3.5 h-3.5 text-[var(--text-dim)]" />
      <span className="text-[var(--text-muted)]">{label}:</span>
      <span className="text-[var(--text-secondary)] font-medium">{value}</span>
    </div>
  );
}
