"use client";

import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ShieldCheck,
  Loader2,
  RotateCcw,
  ChevronDown,
  Clock,
  Info,
  Sparkles,
  Settings2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { VerdictBanner } from "@/components/verification/VerdictBanner";
import { ClaimAnalysisCard } from "@/components/verification/ClaimAnalysisCard";
import { LocalAttributionPanel } from "@/components/verification/LocalAttributionPanel";
import { VerificationTracePanel } from "@/components/verification/VerificationTracePanel";
import { VerificationUnavailable } from "@/components/ui/EmptyState";
import { useAnalysis } from "@/hooks/use-analysis";
import { useAnalysisStore } from "@/store/analysis-store";
import { formatLatency } from "@/lib/format";
import { cn } from "@/lib/utils";
import { getPillar1Diagnostic, getPillar2Diagnostic, getPillar3Diagnostic } from "@/lib/constants";
import type { EvidenceItem } from "@/types/hallucisense";
import { toast } from "sonner";

const SAMPLE_PRESETS = [
  {
    label: "Verified Fact",
    query: "When was James Webb Space Telescope launched?",
    response: "The James Webb Space Telescope was successfully launched into orbit on December 25, 2021 aboard an Ariane 5 rocket from Kourou, French Guiana.",
  },
  {
    label: "Unit Error",
    query: "What is the speed of light?",
    response: "The speed of light in vacuum is approximately 299,792,458 km/s.",
  },
  {
    label: "Temporal Error",
    query: "Tell me about the iPhone launch history.",
    response: "Steve Jobs announced the original iPhone in 2007. Later, Apple launched the iPhone 15 in 1999 with revolutionary AI capabilities.",
  },
  {
    label: "Negation Error",
    query: "Can humans breathe underwater?",
    response: "Humans can breathe underwater without any equipment due to their evolved gill structures.",
  },
];

export default function VerifyPage() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [contextEvidence, setContextEvidence] = useState("");

  const analysis = useAnalysis();
  const currentResult = useAnalysisStore((s) => s.currentResult);
  const isAnalyzing = useAnalysisStore((s) => s.isAnalyzing);
  const reset = useAnalysisStore((s) => s.reset);

  const isLoading = analysis.isPending || isAnalyzing;
  const hasResult = currentResult !== null;
  const hasFailed = analysis.isError;

  const handleVerify = useCallback(async () => {
    const textToVerify = response.trim();
    if (!textToVerify) {
      toast.error("Please enter an LLM response to verify.");
      return;
    }

    reset();

    const providedEvidence: EvidenceItem[] = contextEvidence.trim()
      ? [{
          claim: query.trim() || "provided_context",
          snippet: contextEvidence.trim(),
          source_name: "Provided Context",
          source_url: "",
        }]
      : [];

    let modelToSend = "gpt-4o";
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("hallucisense_default_model");
      if (saved) modelToSend = saved;
    }

    const VALID_FRONTEND_MODELS = [
      "claude", "claude-3-5-sonnet", "deepseek", "default", "gemini",
      "gpt-3.5-turbo", "gpt-4", "gpt-4.1", "gpt-4o", "llama-3",
      "llama-3-70b", "mistral", "phi", "qwen",
    ];
    if (!VALID_FRONTEND_MODELS.includes(modelToSend.toLowerCase())) {
      modelToSend = "gpt-4o";
    }

    analysis.mutate({
      query: query.trim() || undefined,
      response: textToVerify,
      model_name: modelToSend,
      provided_evidence: providedEvidence.length > 0 ? providedEvidence : undefined,
    });
  }, [response, query, contextEvidence, analysis, reset]);

  const handlePreset = (preset: typeof SAMPLE_PRESETS[0]) => {
    setQuery(preset.query);
    setResponse(preset.response);
    reset();
  };

  const handleReset = () => {
    setQuery("");
    setResponse("");
    setContextEvidence("");
    reset();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleVerify();
    }
  };

  return (
    <div className="max-w-[960px] mx-auto p-5 md:p-8 pb-20 md:pb-8 space-y-6">
      {/* ── Page Header ─────────────────────────────────────────────── */}
      <div>
        <h1 className="text-heading-lg text-[var(--text-primary)]">Verify</h1>
        <p className="text-label-md text-[var(--text-muted)] mt-1">
          Analyze any LLM response for hallucinations, factual errors, and inconsistencies.
        </p>
      </div>

      {/* ── Input Section ────────────────────────────────────────────── */}
      {!hasResult && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          {/* Query (optional) */}
          <div>
            <label htmlFor="verify-query" className="block text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1.5">
              Original Question <span className="text-[var(--text-dim)] normal-case tracking-normal">(optional)</span>
            </label>
            <input
              id="verify-query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="What question was asked to the LLM?"
              className={cn(
                "w-full px-3 py-2.5 rounded-[var(--radius-md)]",
                "bg-[var(--bg-surface)] border border-[var(--border)]",
                "text-sm text-[var(--text-primary)] placeholder:text-[var(--text-dim)]",
                "focus:outline-none focus:border-[var(--primary)] focus:ring-1 focus:ring-[var(--primary-soft)]",
                "transition-all duration-150"
              )}
            />
          </div>

          {/* Response (required) */}
          <div>
            <label htmlFor="verify-response" className="block text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1.5">
              LLM Response <span className="text-[var(--hallucination)] normal-case tracking-normal">*</span>
            </label>
            <textarea
              id="verify-response"
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Paste the AI-generated response to verify…"
              rows={6}
              className={cn(
                "w-full px-3 py-3 rounded-[var(--radius-md)] resize-y",
                "bg-[var(--bg-surface)] border border-[var(--border)]",
                "text-sm text-[var(--text-primary)] placeholder:text-[var(--text-dim)]",
                "focus:outline-none focus:border-[var(--primary)] focus:ring-1 focus:ring-[var(--primary-soft)]",
                "transition-all duration-150",
                "min-h-[120px]"
              )}
            />
          </div>

          {/* Advanced options toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-[12px] text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors cursor-pointer"
          >
            <Settings2 className="w-3.5 h-3.5" />
            Advanced options
            <ChevronDown className={cn("w-3 h-3 transition-transform", showAdvanced && "rotate-180")} />
          </button>

          <AnimatePresence>
            {showAdvanced && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="space-y-3 pt-1">
                  <div>
                    <label htmlFor="verify-evidence" className="block text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-1.5">
                      Context / Evidence <span className="text-[var(--text-dim)] normal-case tracking-normal">(optional)</span>
                    </label>
                    <textarea
                      id="verify-evidence"
                      value={contextEvidence}
                      onChange={(e) => setContextEvidence(e.target.value)}
                      placeholder="Optionally provide ground-truth context or evidence for more accurate verification…"
                      rows={3}
                      className={cn(
                        "w-full px-3 py-2.5 rounded-[var(--radius-md)] resize-y",
                        "bg-[var(--bg-surface)] border border-[var(--border)]",
                        "text-sm text-[var(--text-primary)] placeholder:text-[var(--text-dim)]",
                        "focus:outline-none focus:border-[var(--primary)] focus:ring-1 focus:ring-[var(--primary-soft)]",
                        "transition-all duration-150"
                      )}
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Sample Presets */}
          <div className="flex flex-wrap gap-2">
            <span className="text-[11px] text-[var(--text-dim)] self-center mr-1">Try:</span>
            {SAMPLE_PRESETS.map((preset) => (
              <button
                key={preset.label}
                onClick={() => handlePreset(preset)}
                className={cn(
                  "px-2.5 py-1 rounded-[var(--radius-sm)] text-[11px] font-medium",
                  "border border-[var(--border)] text-[var(--text-muted)]",
                  "hover:border-[var(--border-hover)] hover:text-[var(--text-secondary)] hover:bg-[var(--surface-hover)]",
                  "transition-all duration-150 cursor-pointer"
                )}
              >
                {preset.label}
              </button>
            ))}
          </div>

          {/* Submit */}
          <div className="flex items-center gap-3 pt-2">
            <Button
              onClick={handleVerify}
              disabled={isLoading || !response.trim()}
              size="lg"
              className="min-w-[140px]"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing…
                </>
              ) : (
                <>
                  <ShieldCheck className="w-4 h-4" />
                  Verify
                </>
              )}
            </Button>
            <span className="text-[11px] text-[var(--text-dim)]">
              <kbd className="font-mono">⌘</kbd>+<kbd className="font-mono">Enter</kbd>
            </span>
          </div>
        </motion.div>
      )}

      {/* ── Loading State ────────────────────────────────────────────── */}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          <Card>
            <CardContent className="p-6 flex flex-col items-center text-center">
              <div className="w-12 h-12 rounded-[var(--radius-lg)] bg-[var(--ai-soft)] border border-[var(--ai-border)] flex items-center justify-center mb-4">
                <Loader2 className="w-5 h-5 text-[var(--ai)] animate-spin" />
              </div>
              <p className="text-sm font-semibold text-[var(--text-primary)] mb-1">Running verification pipeline</p>
              <p className="text-xs text-[var(--text-muted)]">
                Evidence retrieval → NLI verification → Symbolic checks → Hybrid fusion
              </p>
              <div className="flex gap-2 mt-4">
                {["Retrieving", "Verifying", "Scoring"].map((stage, i) => (
                  <Badge key={stage} variant="ai" size="sm" className={cn("animate-pulse", `stagger-${i + 1}`)}>
                    {stage}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* ── Error State ──────────────────────────────────────────────── */}
      {hasFailed && !hasResult && (
        <VerificationUnavailable onRetry={handleVerify} />
      )}

      {/* ── Results ──────────────────────────────────────────────────── */}
      {hasResult && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-5"
        >
          {/* New Verification Button */}
          <div className="flex items-center justify-between">
            <Button variant="outline" size="sm" onClick={handleReset}>
              <RotateCcw className="w-3.5 h-3.5" />
              New Verification
            </Button>
            <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
              {currentResult.processing_time_ms && (
                <Badge variant="outline" size="sm">
                  <Clock className="w-3 h-3" /> {formatLatency(currentResult.processing_time_ms)}
                </Badge>
              )}
              {currentResult.version && (
                <Badge variant="outline" size="sm">v{currentResult.version}</Badge>
              )}
            </div>
          </div>

          {/* Verdict Banner */}
          <VerdictBanner
            riskLevel={currentResult.risk_level}
            hScore={currentResult.overall_h_score}
            rootCause={currentResult.root_cause_classification}
            traceId={currentResult.trace_id}
            latencyMs={currentResult.processing_time_ms ?? currentResult.latency_ms}
            totalClaims={currentResult.total_sentences_count ?? currentResult.sentence_scores?.length}
            flaggedClaims={currentResult.flagged_sentences_count}
            correctionAvailable={true}
            onCorrect={() => {
              toast.info("Correction feature available through the Chat interface.");
            }}
          />

          {/* Pillar Scores */}
          {currentResult.pillar_scores && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-[var(--text-muted)]" />
                    Three-Pillar Verification Signals
                  </CardTitle>
                  <span className="text-[10px] font-mono text-[var(--text-dim)] uppercase">
                    H = 0.45 FE + 0.30 CG + 0.25 CF
                  </span>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <PillarScoreCard
                    pillarNumber={1}
                    label="Evidence Grounding (FE)"
                    sublabel="Does external evidence support this claim?"
                    diagnosticQuestion="External evidence retrieval and NLI entailment evaluate factual grounding."
                    value={currentResult.pillar_scores.retrieval ?? currentResult.pillar_scores.pillar1_factual_error}
                    status={currentResult.pillar_status?.p1_available !== false ? "active" : "unavailable"}
                    reason="Evidence retrieval service unavailable"
                  />
                  <PillarScoreCard
                    pillarNumber={2}
                    label="Confidence Gap (CG)"
                    sublabel="How uncertain was the model during generation?"
                    diagnosticQuestion="Token log-probability entropy and predictive uncertainty analysis."
                    value={currentResult.pillar_scores.confidence ?? currentResult.pillar_scores.pillar2_confidence_gap}
                    status={currentResult.pillar_status?.p2_available ? "active" : "unavailable"}
                    reason="Token log-probabilities not provided"
                  />
                  <PillarScoreCard
                    pillarNumber={3}
                    label="Consistency Failure (CF)"
                    sublabel="Do independent generations agree with this claim?"
                    diagnosticQuestion="Multi-candidate semantic consistency and cross-sample contradiction analysis."
                    value={currentResult.pillar_scores.consistency ?? currentResult.pillar_scores.pillar3_consistency_failure}
                    status={currentResult.pillar_status?.p3_available ? "active" : "unavailable"}
                    reason="Multiple generations not available"
                  />
                </div>

                {/* Explainability Insight */}
                <div className="mt-3 p-3 rounded-[var(--radius)] bg-black/30 border border-white/[0.04] text-[11px] text-[var(--text-dim)] flex items-start gap-2">
                  <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
                  <p className="leading-relaxed">
                    <strong className="text-[var(--text-secondary)]">Explainable Signal Separation:</strong> A model can be internally confident (<span className="text-emerald-400 font-mono">low CG</span>) and consistent (<span className="text-emerald-400 font-mono">low CF</span>) while remaining factually false (<span className="text-red-400 font-mono">high FE</span>). HalluciSense evaluates all three independent dimensions before computing the calibrated verdict.
                  </p>
                </div>

                {/* Fusion Decomposition */}
                {currentResult.fusion_decomposition && (
                  <div className="mt-3 p-3 rounded-[var(--radius)] bg-[var(--surface)] border border-[var(--border)]">
                    <div className="flex items-center gap-2 mb-2">
                      <Info className="w-3.5 h-3.5 text-[var(--text-dim)]" />
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                        Adaptive Hybrid Fusion Decomposition
                      </span>
                      <Badge variant="ai" size="sm">{currentResult.fusion_decomposition.fusion_mode}</Badge>
                    </div>
                    <p className="text-[12px] text-[var(--text-muted)] font-mono leading-relaxed break-all">
                      {currentResult.fusion_decomposition.equation}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Claim-Level Analysis */}
          {currentResult.sentence_scores && currentResult.sentence_scores.length > 0 && (
            <div>
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-3">
                Claim-Level Analysis ({currentResult.sentence_scores.length} claims)
              </h3>
              <div className="space-y-2">
                {currentResult.sentence_scores.map((claim, i) => (
                  <ClaimAnalysisCard key={i} claim={claim} index={i} />
                ))}
              </div>
            </div>
          )}

          {/* Token Heatmap (if available) */}
          {currentResult.token_heatmap && currentResult.token_heatmap.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm">
                  Token Risk Heatmap
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-1">
                  {currentResult.token_heatmap.map((token, i) => (
                    <span
                      key={i}
                      className={cn(
                        "px-1 py-0.5 rounded text-[12px] font-mono",
                        token.is_hallucination_suspect ? "token-red" :
                          token.tier === "GREEN" ? "token-green" :
                          token.tier === "YELLOW" ? "token-yellow" :
                          token.tier === "ORANGE" ? "token-orange" :
                          token.tier === "RED" ? "token-red" :
                          "text-[var(--text-secondary)]"
                      )}
                      title={`Score: ${token.score?.toFixed(3) ?? "—"}, Entropy: ${token.entropy?.toFixed(3) ?? "—"}`}
                    >
                      {token.token}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Evidence Intelligence Audit Trace */}
          {currentResult.verification_summary && (
            <VerificationTracePanel summary={currentResult.verification_summary} />
          )}

          {/* Local Feature Attribution & Evidence Grounding Panel */}
          {(currentResult.local_attribution || currentResult.semantic_grounding) && (
            <LocalAttributionPanel
              attribution={currentResult.local_attribution}
              semanticGrounding={currentResult.semantic_grounding}
            />
          )}
        </motion.div>
      )}
    </div>
  );
}

function PillarScoreCard({
  pillarNumber,
  label,
  sublabel,
  diagnosticQuestion,
  value,
  status,
  reason,
}: {
  pillarNumber?: 1 | 2 | 3;
  label: string;
  sublabel: string;
  diagnosticQuestion?: string;
  value?: number | null;
  status: "active" | "unavailable";
  reason?: string;
}) {
  const isAvailable = status === "active" && value !== null && value !== undefined;
  const scorePercent = isAvailable ? Math.round(value * 100) : null;
  const riskColor = isAvailable
    ? value > 0.5 ? "var(--hallucination)" : value > 0.25 ? "var(--warning)" : "var(--verified)"
    : "var(--text-dim)";

  const pillarAccent = pillarNumber === 1
    ? "border-teal-400/20"
    : pillarNumber === 2
    ? "border-blue-400/20"
    : pillarNumber === 3
    ? "border-amber-400/20"
    : "border-[var(--border)]";

  const pillarTagColor = pillarNumber === 1
    ? "text-teal-400 bg-teal-500/10 border-teal-500/20"
    : pillarNumber === 2
    ? "text-blue-400 bg-blue-500/10 border-blue-500/20"
    : pillarNumber === 3
    ? "text-amber-400 bg-amber-500/10 border-amber-500/20"
    : "text-[var(--text-dim)] bg-[var(--surface)] border-[var(--border)]";

  // Dynamic interpretation and explicit metric labels
  let metricLabel = "Risk Metric";
  let interpretation = "";

  if (pillarNumber === 1) {
    const diag = getPillar1Diagnostic(value, isAvailable);
    metricLabel = diag.metricLabel;
    interpretation = diag.interpretation;
  } else if (pillarNumber === 2) {
    const diag = getPillar2Diagnostic(value, isAvailable);
    metricLabel = diag.metricLabel;
    interpretation = diag.interpretation;
  } else if (pillarNumber === 3) {
    const diag = getPillar3Diagnostic(value, isAvailable);
    metricLabel = diag.metricLabel;
    interpretation = diag.interpretation;
  }

  return (
    <div className={cn("rounded-[var(--radius)] bg-[var(--surface)] p-3.5 border flex flex-col justify-between space-y-3", pillarAccent)}>
      <div>
        <div className="flex items-center justify-between gap-2 mb-1">
          <p className="text-[12px] font-semibold uppercase tracking-wider text-[var(--text-primary)] truncate" title={diagnosticQuestion || label}>
            {label}
          </p>
          {pillarNumber && (
            <span className={cn("text-[9px] font-mono font-bold px-1.5 py-0.5 rounded border shrink-0", pillarTagColor)}>
              P{pillarNumber}
            </span>
          )}
        </div>
        <p className="text-[10px] text-[var(--text-dim)]">{sublabel}</p>
      </div>

      <div>
        {isAvailable ? (
          <div className="space-y-2">
            <div className="flex items-baseline gap-1.5">
              <span className="text-[11px] font-semibold text-[var(--text-muted)] font-mono">{metricLabel}:</span>
              <span
                className="text-xl font-bold font-mono"
                style={{ color: riskColor }}
              >
                {scorePercent}%
              </span>
            </div>
            <div className="pt-2 border-t border-[var(--border)]">
              <span className="text-[9px] font-semibold uppercase tracking-wider text-[var(--text-dim)] block mb-0.5">
                Interpretation:
              </span>
              <p className="text-[11px] text-[var(--text-secondary)] font-medium leading-snug">
                {interpretation}
              </p>
            </div>
          </div>
        ) : (
          <div>
            <p className="text-sm text-[var(--text-dim)] italic">Unavailable</p>
            {reason && (
              <p className="text-[10px] text-[var(--text-dim)] mt-1 leading-tight">{reason}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
