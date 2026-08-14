"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Zap,
  ArrowRight,
  GitBranch,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Sparkles,
  Layers,
  Activity,
  Database,
  RefreshCw,
  Sliders,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, GlassCard } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { InlineError } from "@/components/ui/InlineError";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useAnalysis } from "@/hooks/use-analysis";
import { getRiskColor, getRiskLabel, formatLatency } from "@/lib/format";
import type { AnalysisResponse } from "@/types/hallucisense";

export default function AnalyzePage() {
  const [activeTab, setActiveTab] = useState("compare");

  // Response A State
  const [responseA, setResponseA] = useState("");
  const [modelA, setModelA] = useState("GPT-4o");

  // Response B State
  const [responseB, setResponseB] = useState("");
  const [modelB, setModelB] = useState("Llama-3-70B");

  // Shared Context & Query
  const [query, setQuery] = useState("");
  const [evidenceText, setEvidenceText] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Analysis Mutations
  const analyzeA = useAnalysis();
  const analyzeB = useAnalysis();

  const [resultA, setResultA] = useState<AnalysisResponse | null>(null);
  const [resultB, setResultB] = useState<AnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<unknown>(null);
  const [showErrorDetails, setShowErrorDetails] = useState(false);

  const handleCompare = async () => {
    if (!responseA.trim() || !responseB.trim()) return;

    setIsAnalyzing(true);
    setErrorMsg(null);
    setErrorDetails(null);
    setShowErrorDetails(false);

    const providedEvidence = evidenceText.trim()
      ? evidenceText.split("\n").filter((line) => line.trim().length > 0)
      : [];

    try {
      const [resA, resB] = await Promise.all([
        analyzeA.mutateAsync({
          text: responseA,
          query,
          provided_evidence: providedEvidence,
          model_name: modelA,
        }),
        analyzeB.mutateAsync({
          text: responseB,
          query,
          provided_evidence: providedEvidence,
          model_name: modelB,
        }),
      ]);

      setResultA(resA);
      setResultB(resB);
    } catch (err: unknown) {
      if (err && typeof err === "object" && "body" in err) {
        const apiErr = err as { message?: string; body?: { details?: unknown } };
        setErrorMsg(apiErr.message || "Invalid request payload schema or missing required fields.");
        setErrorDetails(apiErr.body?.details || apiErr.body || null);
      } else {
        setErrorMsg(err instanceof Error ? err.message : "Failed to complete comparative response analysis.");
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClear = () => {
    setResponseA("");
    setResponseB("");
    setQuery("");
    setEvidenceText("");
    setResultA(null);
    setResultB(null);
    setErrorMsg(null);
  };

  return (
    <div className="p-6 md:p-8 pl-12 md:pl-16 space-y-8 max-w-7xl mx-auto">
      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.04] pb-6">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/[0.04] bg-white/[0.02] text-slate-400 text-xs font-semibold tracking-wide uppercase mb-2 font-mono">
            <Zap className="w-3.5 h-3.5" />
            Comparative Intelligence
          </div>
          <h1 className="text-heading-md font-bold text-white tracking-tight">
            Response Divergence & Comparison Analyzer
          </h1>
          <p className="text-label-md text-slate-400 mt-1 max-w-2xl">
            Side-by-side factual verification, logit uncertainty divergence, and semantic contradiction analysis across LLM candidates.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={handleClear}
            className="border-white/5 text-slate-300 hover:text-white font-mono text-xs cursor-pointer"
          >
            Clear Fields
          </Button>
        </div>
      </div>

      {/* ── Workspace Mode Tabs ─────────────────────────────────────────── */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="bg-bg-surface border border-white/[0.04] p-1 rounded-xl">
          <TabsTrigger value="compare">Compare Responses</TabsTrigger>
          <TabsTrigger value="batch">Batch Analysis</TabsTrigger>
          <TabsTrigger value="divergence">Semantic Divergence</TabsTrigger>
        </TabsList>

        {/* ── MODE 1: COMPARE RESPONSES ───────────────────────────────────── */}
        <TabsContent value="compare" className="space-y-6">
          {/* Input Grid: Model A vs Model B (stacked below 1024px) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Candidate A */}
            <Card className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                  <span className="text-sm font-semibold text-white">Candidate Output A</span>
                </div>
                <Input
                  value={modelA}
                  onChange={(e) => setModelA(e.target.value)}
                  placeholder="Model name (e.g. GPT-4o)"
                  className="w-36 h-8 text-xs font-mono bg-white/[0.01] border-white/5 focus:border-accent-primary/40 text-slate-300"
                />
              </div>

              <Textarea
                value={responseA}
                onChange={(e) => setResponseA(e.target.value)}
                placeholder="Paste first model response output here..."
                rows={6}
                className="bg-black/30 font-mono text-sm text-slate-200 border-white/[0.04] focus:border-accent-primary/40"
              />

              <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                <span>{responseA.length} chars</span>
                <span>{responseA.trim() ? responseA.trim().split(/\s+/).length : 0} words</span>
              </div>
            </Card>

            {/* Candidate B */}
            <Card className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-indigo-500" />
                  <span className="text-sm font-semibold text-white">Candidate Output B</span>
                </div>
                <Input
                  value={modelB}
                  onChange={(e) => setModelB(e.target.value)}
                  placeholder="Model name (e.g. Llama-3-70B)"
                  className="w-36 h-8 text-xs font-mono bg-white/[0.01] border-white/5 focus:border-accent-primary/40 text-slate-300"
                />
              </div>

              <Textarea
                value={responseB}
                onChange={(e) => setResponseB(e.target.value)}
                placeholder="Paste second model response output here to compare..."
                rows={6}
                className="bg-black/30 font-mono text-sm text-slate-200 border-white/[0.04] focus:border-accent-primary/40"
              />

              <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                <span>{responseB.length} chars</span>
                <span>{responseB.trim() ? responseB.trim().split(/\s+/).length : 0} words</span>
              </div>
            </Card>
          </div>

          {/* Optional Shared Context Drawer */}
          <Card className="overflow-hidden">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-slate-400 hover:text-white transition-colors cursor-pointer font-mono"
            >
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-slate-400" />
                <span>Shared Grounding Context & User Query (Optional)</span>
              </div>
              {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showAdvanced && (
              <div className="p-4 border-t border-white/[0.04] space-y-4">
                <div>
                  <label className="text-label-sm text-slate-400 block mb-1">User Query</label>
                  <Input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. What were the key events of Apollo 11?"
                    className="bg-bg-surface border-white/[0.04] focus:border-accent-primary/40 text-slate-300 text-xs font-mono"
                  />
                </div>
                <div>
                  <label className="text-label-sm text-slate-400 block mb-1">Reference Evidence Passages</label>
                  <Textarea
                    value={evidenceText}
                    onChange={(e) => setEvidenceText(e.target.value)}
                    placeholder="Enter reference sentences (one per line)..."
                    rows={3}
                    className="bg-bg-surface border-white/[0.04] focus:border-accent-primary/40 text-slate-300 text-xs font-mono"
                  />
                </div>
              </div>
            )}
          </Card>

          {/* Contextual Inline Error Feedback */}
          {errorMsg && (
            <InlineError
              message={errorMsg}
              details={errorDetails}
              onClear={() => setErrorMsg(null)}
              className="mt-2"
            />
          )}

          {/* Action CTA */}
          <div className="flex items-center justify-between">
            <div className="text-xs text-slate-500 max-w-md font-mono">
              Executes parallel NLI claim grounding, logit entropy, and self-consistency analysis on both outputs.
            </div>

            <Button
              onClick={handleCompare}
              disabled={isAnalyzing || !responseA.trim() || !responseB.trim()}
              className="bg-accent-primary hover:bg-accent-primary/90 text-white font-semibold shadow-[0_0_24px_rgba(168,85,247,0.2)] rounded-xl cursor-pointer font-mono"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Analyzing Candidates...
                </>
              ) : (
                <>
                  Compare Candidates
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </Button>
          </div>

          {/* Loading Skeleton State */}
          {isAnalyzing && (
            <div className="space-y-6 pt-6 border-t border-white/[0.04] animate-pulse">
              <div className="h-6 w-48 bg-white/5 rounded mb-4" />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="h-48 bg-bg-surface/60 border border-white/5 rounded-2xl" />
                <div className="h-48 bg-bg-surface/60 border border-white/5 rounded-2xl" />
              </div>
              <div className="h-36 bg-bg-surface/60 border border-white/5 rounded-2xl" />
            </div>
          )}

          {/* ── COMPARATIVE RESULTS PANEL ─────────────────────────────────── */}
          {resultA && resultB && !isAnalyzing && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6 pt-6 border-t border-white/[0.08]"
            >
              <h2 className="text-heading-md font-bold text-white tracking-tight">Comparative Verification Results</h2>

              {/* Side-by-side Score Cards */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <CompareScoreCard result={resultA} modelName={modelA} accentColor="blue" />
                <CompareScoreCard result={resultB} modelName={modelB} accentColor="indigo" />
              </div>

              {/* Pillar Score Breakdown Table */}
              <Card className="p-6 space-y-4">
                <h3 className="text-sm font-semibold text-slate-300">Three-Pillar Risk Divergence</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b border-white/10 text-slate-400">
                        <th className="pb-2">Verification Pillar</th>
                        <th className="pb-2 text-slate-200">{modelA}</th>
                        <th className="pb-2 text-slate-200">{modelB}</th>
                        <th className="pb-2 text-right">Divergence Delta</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      <PillarDivergenceRow
                        label="Pillar 1: Factual Error"
                        valA={resultA.pillar_scores.retrieval ?? 0}
                        valB={resultB.pillar_scores.retrieval ?? 0}
                      />
                      <PillarDivergenceRow
                        label="Pillar 2: Confidence Gap"
                        valA={resultA.pillar_scores.confidence ?? 0}
                        valB={resultB.pillar_scores.confidence ?? 0}
                      />
                      <PillarDivergenceRow
                        label="Pillar 3: Consistency Failure"
                        valA={resultA.pillar_scores.consistency ?? 0}
                        valB={resultB.pillar_scores.consistency ?? 0}
                      />
                      <PillarDivergenceRow
                        label="Overall H-Score"
                        valA={resultA.overall_h_score}
                        valB={resultB.overall_h_score}
                        isBold
                      />
                    </tbody>
                  </table>
                </div>
              </Card>
            </motion.div>
          )}
        </TabsContent>

        {/* ── MODE 2: BATCH ANALYSIS ─────────────────────────────────────── */}
        <TabsContent value="batch">
          <div className="space-y-4">
            <Card className="p-12 border-dashed border-white/10 bg-white/[0.01] hover:bg-white/[0.02] transition-colors text-center space-y-4 flex flex-col items-center justify-center cursor-pointer group">
              <div className="w-12 h-12 rounded-full bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-slate-400 group-hover:text-white transition-colors">
                <Layers className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-bold text-white">Upload Batch Evaluation Dataset</h3>
                <p className="text-xs text-slate-400 max-w-sm mx-auto leading-relaxed">
                  Drag and drop your evaluation dataset (CSV or JSON format, max 50MB) or click to browse local files.
                </p>
              </div>
              <Button size="sm" variant="outline" className="border-white/10 hover:bg-white/[0.04] text-xs font-mono">
                Select File
              </Button>
            </Card>
            <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-white/[0.02] border border-white/[0.04] text-[11px] text-slate-500 font-mono">
              <span className="text-accent-primary font-bold">ℹ NOTE:</span>
              <span>Bulk evaluation executes asynchronously and logs individual results directly to Pipeline Traces.</span>
            </div>
          </div>
        </TabsContent>

        {/* ── MODE 3: CLAIM DIVERGENCE ───────────────────────────────────── */}
        <TabsContent value="divergence">
          <div className="space-y-4">
            <Card className="p-12 text-center space-y-4 flex flex-col items-center justify-center">
              <div className="w-12 h-12 rounded-full bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-slate-400">
                <GitBranch className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-bold text-white">Semantic Claim Divergence Matrix</h3>
                <p className="text-xs text-slate-400 max-w-sm mx-auto leading-relaxed">
                  Enter candidate outputs in the Compare Responses tab first to map cross-model claim alignment and contradiction heatmaps.
                </p>
              </div>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

/* ── Helper Components ────────────────────────────────────────────────────── */

function CompareScoreCard({
  result,
  modelName,
  accentColor,
}: {
  result: AnalysisResponse;
  modelName: string;
  accentColor: "blue" | "indigo";
}) {
  const hPct = (result.overall_h_score * 100).toFixed(1);
  const colorHex = getRiskColor(result.risk_level);

  return (
    <Card className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono font-bold text-white tracking-wide uppercase">{modelName}</span>
        <StatusBadge
          label={getRiskLabel(result.risk_level)}
          status={
            result.risk_level === "VERIFIED"
              ? "success"
              : result.risk_level === "LIKELY_HALLUCINATED"
              ? "error"
              : "warning"
          }
        />
      </div>

      <div className="space-y-1">
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-slate-400 font-mono">Hallucination Risk Score</span>
          <span className="text-2xl font-bold font-mono" style={{ color: colorHex }}>
            {hPct}%
          </span>
        </div>
        <div className="h-2 rounded-full bg-white/10 overflow-hidden">
          <div className="h-full rounded-full transition-all duration-500" style={{ width: `${hPct}%`, backgroundColor: colorHex }} />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-white/5 text-[11px] font-mono">
        <div>
          <span className="text-slate-500 block font-sans">Grounding</span>
          <span className="text-slate-200 font-semibold">{((result.pillar_scores.retrieval ?? 0) * 100).toFixed(0)}%</span>
        </div>
        <div>
          <span className="text-slate-500 block font-sans">Confidence</span>
          <span className="text-slate-200 font-semibold">{((result.pillar_scores.confidence ?? 0) * 100).toFixed(0)}%</span>
        </div>
        <div>
          <span className="text-slate-500 block font-sans">Consistency</span>
          <span className="text-slate-200 font-semibold">{((result.pillar_scores.consistency ?? 0) * 100).toFixed(0)}%</span>
        </div>
      </div>
    </Card>
  );
}

function PillarDivergenceRow({
  label,
  valA,
  valB,
  isBold = false,
}: {
  label: string;
  valA: number;
  valB: number;
  isBold?: boolean;
}) {
  const delta = (valA - valB) * 100;
  const deltaStr = `${delta > 0 ? "+" : ""}${delta.toFixed(1)}%`;

  return (
    <tr className={isBold ? "font-bold text-white" : "text-slate-300"}>
      <td className="py-2.5 font-sans">{label}</td>
      <td className="py-2.5 text-slate-300 font-mono">{(valA * 100).toFixed(1)}%</td>
      <td className="py-2.5 text-slate-300 font-mono">{(valB * 100).toFixed(1)}%</td>
      <td className={`py-2.5 text-right font-mono ${delta > 0 ? "text-status-error" : delta < 0 ? "text-status-success" : "text-slate-500"}`}>
        {deltaStr}
      </td>
    </tr>
  );
}
