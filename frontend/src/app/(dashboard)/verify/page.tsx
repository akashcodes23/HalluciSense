"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ShieldCheck,
  Loader2,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Clock,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { InlineError } from "@/components/ui/InlineError";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useAnalysis, useExplain } from "@/hooks/use-analysis";
import { useAnalysisStore } from "@/store/analysis-store";
import { formatLatency, getRiskColor, getRiskLabel } from "@/lib/format";
import { ScoreGauge } from "@/components/features/analyzer/score-gauge";
import { TokenHeatmap } from "@/components/features/heatmap/token-heatmap";
import type { AnalysisResponse, SentenceScore, EvidenceItem } from "@/types/hallucisense";
import { toast } from "sonner";

const SAMPLE_PRESETS = [
  {
    label: "Temporal Verification",
    query: "When was James Webb Space Telescope launched?",
    response: "The James Webb Space Telescope was successfully launched into orbit on December 25, 2021 aboard an Ariane 5 rocket from Kourou, French Guiana.",
  },
  {
    label: "Adversarial Date Contamination",
    query: "Tell me about the iPhone launch history.",
    response: "Steve Jobs announced the original iPhone in 2007. Later, Apple launched the iPhone 15 in 1999 with revolutionary AI capabilities.",
  },
  {
    label: "Epistemic Protection (Prediction)",
    query: "What will happen in quantum computing by 2030?",
    response: "We predict that fault-tolerant quantum computers might achieve commercial quantum supremacy for drug discovery before 2030.",
  },
];

export default function VerifyPage() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [contextEvidence, setContextEvidence] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [validationError, setValidationError] = useState<{ message: string; details?: unknown } | null>(null);

  const analysis = useAnalysis();
  const explain = useExplain();
  const currentResult = useAnalysisStore((s) => s.currentResult);
  const currentExplain = useAnalysisStore((s) => s.currentExplain);
  const isAnalyzing = useAnalysisStore((s) => s.isAnalyzing);
  const reset = useAnalysisStore((s) => s.reset);

  const isLoading = analysis.isPending || isAnalyzing;

  const handleVerify = async () => {
    const textToVerify = response.trim();
    if (!textToVerify) {
      toast.error("Please enter an LLM response to verify.");
      return;
    }

    reset();
    setValidationError(null);

    const providedEvidence: EvidenceItem[] = contextEvidence.trim()
      ? [
          {
            claim: query.trim() || "provided_context",
            snippet: contextEvidence.trim(),
            source_name: "Provided Context",
            source_url: "",
          },
        ]
      : [];

    let modelToSend = "gpt-4o";
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("hallucisense_default_model");
      if (saved) {
        modelToSend = saved;
      }
    }

    const VALID_FRONTEND_MODELS = [
      "claude",
      "claude-3-5-sonnet",
      "deepseek",
      "default",
      "gemini",
      "gpt-3.5-turbo",
      "gpt-4",
      "gpt-4.1",
      "gpt-4o",
      "llama-3",
      "llama-3-70b",
      "mistral",
      "phi",
      "qwen"
    ];

    const normalizedModel = modelToSend.trim().toLowerCase();
    const isValid = VALID_FRONTEND_MODELS.includes(normalizedModel) ||
                    ["gpt", "gemini", "claude", "llama", "qwen", "mistral"].some(m => normalizedModel.includes(m));

    if (!isValid) {
      setValidationError({
        message: `Validation Error: Invalid or unsupported model name "${modelToSend}". Supported options include: ${VALID_FRONTEND_MODELS.join(", ")}.`
      });
      toast.error(`Invalid model "${modelToSend}" selected in settings.`);
      return;
    }

    const payload = {
      text: textToVerify,
      response: textToVerify,
      query: query.trim(),
      provided_evidence: providedEvidence,
      model_name: modelToSend,
    };

    try {
      await analysis.mutateAsync(payload);
      explain.mutate(payload);
      toast.success("Verification complete");
    } catch (err: unknown) {
      if (err && typeof err === "object" && "body" in err) {
        const apiErr = err as { message?: string; body?: { details?: unknown } };
        setValidationError({
          message: apiErr.message || "Invalid request payload schema or missing required fields.",
          details: apiErr.body?.details || apiErr.body || null,
        });
      } else {
        const message = err instanceof Error ? err.message : "Verification failed";
        setValidationError({ message });
      }
    }
  };

  const handleReset = () => {
    setQuery("");
    setResponse("");
    setContextEvidence("");
    setValidationError(null);
    reset();
  };

  const applyPreset = (preset: typeof SAMPLE_PRESETS[0]) => {
    setQuery(preset.query);
    setResponse(preset.response);
    setContextEvidence("");
    setValidationError(null);
    reset();
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-8 space-y-8">
        {/* ── Page Title & Subtitle ───────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/[0.04] pb-6"
        >
          <div>
            <div className="flex items-center gap-2 mb-1">
              <ShieldCheck className="w-6 h-6 text-slate-400" />
              <h1 className="text-heading-md font-bold text-white tracking-tight">Verification Workspace</h1>
            </div>
            <p className="text-label-md text-slate-400 max-w-xl">
              Determine whether an AI-generated response is factual, temporally consistent, and grounded using confidence-aware hybrid verification.
            </p>
          </div>

          {/* Quick Preset Buttons */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-label-sm text-slate-500 font-medium mr-1 font-mono">Presets:</span>
            {SAMPLE_PRESETS.map((p, idx) => (
              <button
                key={idx}
                onClick={() => applyPreset(p)}
                disabled={isLoading}
                className="px-2.5 py-1 text-[11px] rounded-lg border border-white/[0.04] bg-white/[0.01] text-slate-300 hover:text-white hover:bg-white/[0.04] transition-colors cursor-pointer disabled:opacity-50 font-mono"
              >
                {p.label}
              </button>
            ))}
          </div>
        </motion.div>

        {/* ── Response & Evidence Input Experience ────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="space-y-4"
        >
          {/* Response Textarea */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label htmlFor="verify-response" className="text-label-sm text-slate-400 font-sans">
                LLM Response to Verify <span className="text-status-error">*</span>
              </label>
              <span className="text-label-sm text-slate-500 font-mono">
                {response.length} chars • {response.trim() ? response.trim().split(/\s+/).length : 0} words
              </span>
            </div>
            <Textarea
              id="verify-response"
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              placeholder="Paste the AI-generated text or claim to evaluate..."
              className="min-h-[140px] text-sm leading-relaxed bg-bg-surface border-white/[0.04] focus:border-accent-primary/40 font-mono text-slate-300"
              disabled={isLoading}
            />
          </div>

          {/* Advanced Context Toggle */}
          <div className="flex items-center justify-between pt-1">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors cursor-pointer font-mono"
            >
              {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              {showAdvanced ? "Hide Optional Query & Context" : "Add Optional User Query & Reference Context"}
            </button>
          </div>

          {/* Optional Inputs Drawer */}
          <AnimatePresence>
            {showAdvanced && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
                className="space-y-4 pt-2 border-t border-white/[0.04]"
              >
                <div className="space-y-2">
                  <label htmlFor="verify-query" className="text-label-sm text-slate-400 font-sans">
                    Original Prompt / Question (Optional)
                  </label>
                  <Textarea
                    id="verify-query"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. When was James Webb Space Telescope launched?"
                    className="min-h-[70px] text-sm bg-bg-surface border-white/[0.04] focus:border-accent-primary/40 font-mono text-slate-300"
                    disabled={isLoading}
                  />
                </div>

                <div className="space-y-2">
                  <label htmlFor="verify-context" className="text-label-sm text-slate-400 font-sans">
                    Reference Evidence / Context Excerpt (Optional)
                  </label>
                  <Textarea
                    id="verify-context"
                    value={contextEvidence}
                    onChange={(e) => setContextEvidence(e.target.value)}
                    placeholder="Paste reference text or ground-truth document against which to verify..."
                    className="min-h-[90px] text-sm bg-bg-surface border-white/[0.04] focus:border-accent-primary/40 font-mono text-slate-300"
                    disabled={isLoading}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Contextual Inline Error Feedback */}
          {validationError && (
            <InlineError
              message={validationError.message}
              details={validationError.details}
              onClear={() => setValidationError(null)}
              className="mt-2"
            />
          )}

          {/* Action Row */}
          <div className="flex items-center justify-between pt-2">
            <div>
              {(response || query || currentResult) && (
                <Button variant="ghost" size="sm" onClick={handleReset} disabled={isLoading} className="text-slate-400 hover:text-slate-200">
                  <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
                  Clear Workspace
                </Button>
              )}
            </div>

            <Button
              onClick={handleVerify}
              disabled={isLoading || !response.trim()}
              size="lg"
              className="min-w-[180px] bg-indigo-600 hover:bg-indigo-500 text-white font-semibold shadow-lg shadow-indigo-600/20 cursor-pointer"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Verifying...
                </>
              ) : (
                <>
                  <ShieldCheck className="w-4 h-4 mr-2" />
                  Verify Response
                </>
              )}
            </Button>
          </div>
        </motion.div>

        {/* ── Verification Progress Indicator ──────────────────────────── */}
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-6 rounded-2xl border border-blue-500/20 bg-blue-500/[0.04] space-y-4"
            >
              <div className="flex items-center gap-3">
                <Loader2 className="w-5 h-5 text-blue-400 animate-spin shrink-0" />
                <div>
                  <h3 className="text-sm font-semibold text-white">Running Verification Pipeline</h3>
                  <p className="text-xs text-slate-400">Extracting atomic claims, querying retrieval indices, resolving epistemic modality, and verifying temporal anchors...</p>
                </div>
              </div>

              {/* Progress Steps */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-2 border-t border-white/[0.06]">
                <ProgressStep label="1. Claim Segmentation" active />
                <ProgressStep label="2. Evidence Alignment" active />
                <ProgressStep label="3. Epistemic Gating" active />
                <ProgressStep label="4. Score Fusion" active />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Result Experience & Progressive Disclosure ─────────────── */}
        <AnimatePresence>
          {currentResult && !isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.4 }}
              className="space-y-6"
            >
              {/* 1. Primary Verdict Header */}
              <Card className="p-6 md:p-8 relative overflow-hidden border-white/[0.04]">
                <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                  {/* Gauge */}
                  <div className="flex items-center gap-6">
                    <ScoreGauge score={currentResult.overall_h_score} riskLevel={currentResult.risk_level} />
                    <div>
                      <div className="flex items-center gap-2 mb-1.5">
                        <RiskBadge level={currentResult.risk_level} />
                        {currentResult.latency_ms && (
                          <span className="text-xs text-slate-500 font-mono flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {formatLatency(currentResult.latency_ms)}
                          </span>
                        )}
                      </div>

                      <h2 className="text-heading-md font-bold text-white tracking-tight">
                        Verdict: {getRiskLabel(currentResult.risk_level)}
                      </h2>
                      <p className="text-label-md text-slate-400 mt-1 max-w-md">
                        Overall Hallucination Index (H-Score):{" "}
                        <span className="font-mono font-semibold" style={{ color: getRiskColor(currentResult.risk_level) }}>
                          {(currentResult.overall_h_score * 100).toFixed(1)}%
                        </span>
                        {" • "}
                        {currentResult.flagged_sentences_count ?? 0} of {currentResult.total_sentences_count ?? (currentResult.sentence_scores?.length || 1)} claims flagged.
                      </p>
                    </div>
                  </div>

                  {/* Summary Meta Pills */}
                  <div className="flex flex-wrap md:flex-col gap-2 shrink-0 border-t md:border-t-0 md:border-l border-white/[0.04] pt-4 md:pt-0 md:pl-6">
                    <MetaPill label="Pillar 1 Factual Error" value={`${((currentResult.pillar_scores?.pillar1_factual_error ?? currentResult.pillar_scores?.retrieval ?? 0) * 100).toFixed(1)}%`} />
                    <MetaPill label="Pillar 2 Confidence Gap" value={currentResult.pillar_scores?.pillar2_confidence_gap != null ? `${(currentResult.pillar_scores.pillar2_confidence_gap * 100).toFixed(1)}%` : "N/A (Protected)"} />
                    <MetaPill label="Pillar 3 Consistency" value={currentResult.pillar_scores?.pillar3_consistency_failure != null ? `${(currentResult.pillar_scores.pillar3_consistency_failure * 100).toFixed(1)}%` : "N/A"} />
                  </div>
                </div>
              </Card>

              {/* 2. Progressive Disclosure Tab View */}
              <Tabs defaultValue="claims" className="space-y-6">
                <TabsList className="bg-bg-surface border border-white/[0.04] p-1 rounded-xl">
                  <TabsTrigger value="claims" className="cursor-pointer">Claim Breakdown ({currentResult.sentence_scores?.length || 0})</TabsTrigger>
                  <TabsTrigger value="evidence" className="cursor-pointer">Evidence Citations ({currentResult.evidence?.length || 0})</TabsTrigger>
                  <TabsTrigger value="technical" className="cursor-pointer">Technical Traces</TabsTrigger>
                  {currentExplain && <TabsTrigger value="explanation" className="cursor-pointer">Explanation & Remediation</TabsTrigger>}
                </TabsList>

                {/* Claim-Level Results Tab */}
                <TabsContent value="claims" className="space-y-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-slate-300">Atomic Claims & Epistemic Modality Analysis</h3>
                    <span className="text-xs text-slate-500">Atomic sentence segmentation & temporal anchor evaluation</span>
                  </div>

                  {currentResult.sentence_scores?.map((sentence, idx) => (
                    <ClaimCard key={idx} sentence={sentence} index={idx} />
                  ))}
                </TabsContent>

                {/* Evidence Citations Tab */}
                <TabsContent value="evidence" className="space-y-4">
                  {!currentResult.evidence || currentResult.evidence.length === 0 ? (
                    <div className="text-center py-12 text-slate-500 text-sm bg-white/[0.02] rounded-xl border border-white/[0.06]">
                      No explicit external evidence passages retrieved for this query.
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {currentResult.evidence.map((ev, i) => (
                        <div key={i} className="p-4 rounded-xl border border-white/[0.06] bg-[#0b1220] space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-semibold text-blue-400">{ev.source_name || ev.source || "Wikipedia/Retriever"}</span>
                            {ev.similarity_score != null && (
                              <span className="text-xs font-mono text-slate-500">
                                Match: {(ev.similarity_score * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-slate-200 leading-relaxed">&quot;{ev.snippet}&quot;</p>
                          {ev.source_url && (
                            <a href={ev.source_url} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-400 hover:underline flex items-center gap-1">
                              View Source <ExternalLink className="w-3 h-3" />
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </TabsContent>

                {/* Technical Traces Tab */}
                <TabsContent value="technical" className="space-y-6">
                  {/* Token Heatmap */}
                  {currentResult.token_heatmap && currentResult.token_heatmap.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="text-sm font-semibold text-slate-300">Token-Level Hallucination Heatmap</h4>
                      <TokenHeatmap tokens={currentResult.token_heatmap} />
                    </div>
                  )}

                  {/* Metadata & Root Cause */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-xl border border-white/[0.06] bg-[#0b1220]">
                      <span className="text-xs text-slate-500 uppercase tracking-wider block mb-1">Root Cause Classification</span>
                      <span className="text-sm font-semibold text-white">{currentResult.root_cause_classification || "None (Grounded)"}</span>
                    </div>
                    <div className="p-4 rounded-xl border border-white/[0.06] bg-[#0b1220]">
                      <span className="text-xs text-slate-500 uppercase tracking-wider block mb-1">Trace Identifier</span>
                      <span className="text-sm font-mono text-slate-300">{currentResult.trace_id || "LOCAL_EXECUTION"}</span>
                    </div>
                    <div className="p-4 rounded-xl border border-white/[0.06] bg-[#0b1220]">
                      <span className="text-xs text-slate-500 uppercase tracking-wider block mb-1">Fusion Weights (α / β / γ)</span>
                      <span className="text-sm font-mono text-slate-300">0.45 / 0.30 / 0.25</span>
                    </div>
                  </div>
                </TabsContent>

                {/* Explanation Tab */}
                {currentExplain && (
                  <TabsContent value="explanation" className="space-y-4">
                    <div className="p-6 rounded-xl border border-white/[0.06] bg-[#0b1220] space-y-4">
                      <h4 className="text-sm font-semibold text-white">Natural Language Reasoning</h4>
                      <p className="text-sm text-slate-300 whitespace-pre-line leading-relaxed">
                        {currentExplain.explanation_markdown || currentExplain.confidence_explanation}
                      </p>

                      {currentExplain.remediation_suggestions && currentExplain.remediation_suggestions.length > 0 && (
                        <div className="pt-4 border-t border-white/[0.06] space-y-2">
                          <h5 className="text-xs font-semibold uppercase tracking-wider text-amber-400">Remediation Suggestions</h5>
                          <ul className="list-disc list-inside text-xs text-slate-400 space-y-1">
                            {currentExplain.remediation_suggestions.map((s, i) => (
                              <li key={i}>{s}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </TabsContent>
                )}
              </Tabs>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

/* ── Auxiliary Components ─────────────────────────────────────────────────── */

function ProgressStep({ label, active }: { label: string; active?: boolean }) {
  return (
    <div className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 ${
      active ? "border-blue-500/30 bg-blue-500/10 text-blue-300" : "border-white/[0.06] text-slate-500"
    }`}>
      <div className={`w-1.5 h-1.5 rounded-full ${active ? "bg-blue-400 animate-ping" : "bg-slate-600"}`} />
      {label}
    </div>
  );
}

function RiskBadge({ level }: { level: string }) {
  const label = getRiskLabel(level);
  const status = level === "VERIFIED" ? "success" : level === "LIKELY_HALLUCINATED" ? "error" : "warning";
  return <StatusBadge label={label} status={status} />;
}

function MetaPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 text-xs font-mono">
      <span className="text-slate-500">{label}:</span>
      <span className="font-semibold text-slate-200">{value}</span>
    </div>
  );
}

function ClaimCard({ sentence, index }: { sentence: SentenceScore; index: number }) {
  const [open, setOpen] = useState(true);
  const score = sentence.h_score ?? sentence.score ?? 0;
  const riskColor = getRiskColor(sentence.risk_level);
  const isProtected = sentence.epistemic_category && sentence.epistemic_category !== "ASSERTED_FACT";

  return (
    <Card className="overflow-hidden transition-colors">
      <div
        onClick={() => setOpen(!open)}
        className="p-4 flex items-start gap-4 cursor-pointer hover:bg-white/[0.01] transition-colors"
      >
        <span className="text-xs font-mono text-slate-500 mt-0.5 shrink-0">#{String(index + 1).padStart(2, "0")}</span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-white leading-snug">{sentence.sentence_text || sentence.text}</p>
          <div className="flex items-center gap-2 mt-2 flex-wrap">
            <RiskBadge level={sentence.risk_level} />

            {sentence.epistemic_category && (
              <StatusBadge
                label={sentence.epistemic_category + (isProtected ? " (Protected Gate)" : "")}
                status={isProtected ? "info" : "default"}
              />
            )}

            {sentence.temporal_anchor?.asserted_year && (
              <span className="text-[10px] font-mono text-slate-400 bg-white/[0.04] px-2 py-0.5 rounded border border-white/[0.06]">
                Year: {sentence.temporal_anchor.asserted_year}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <div className="text-right">
            <span className="text-xs font-mono font-bold block" style={{ color: riskColor }}>
              {(score * 100).toFixed(0)}%
            </span>
            <span className="text-[10px] text-slate-500">H-Score</span>
          </div>
          {open ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
        </div>
      </div>

      {open && (
        <div className="px-4 pb-4 pt-2 border-t border-white/[0.04] bg-black/20 space-y-3 text-xs">
          {sentence.reasoning_summary && (
            <p className="text-slate-300 leading-relaxed">
              <strong className="text-slate-400">Reasoning:</strong> {sentence.reasoning_summary}
            </p>
          )}

          {sentence.nli_entailment_prob != null && (
            <div className="flex items-center gap-4 text-slate-400 font-mono">
              <span>Entailment: {(sentence.nli_entailment_prob * 100).toFixed(1)}%</span>
              <span>Contradiction: {((sentence.nli_contradiction_prob || 0) * 100).toFixed(1)}%</span>
            </div>
          )}

          {sentence.evidence_matched && sentence.evidence_matched.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-500 block">Matched Evidence Excerpt</span>
              {sentence.evidence_matched.map((ev, ei) => (
                <div key={ei} className="p-2.5 rounded bg-white/[0.03] border border-white/[0.06] text-slate-300">
                  &quot;{ev.snippet}&quot; — <span className="text-blue-400">{ev.source_name || "Wikipedia"}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
