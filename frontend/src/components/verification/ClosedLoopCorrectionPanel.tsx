"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ArrowRight,
  ShieldCheck,
  GitBranch,
  Copy,
  Check,
  BookOpen,
  Info,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { correctResponse } from "@/services/hallucisense-api";
import type { AnalysisResponse, CorrectionResponse } from "@/types/hallucisense";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import Link from "next/link";

interface ClosedLoopCorrectionPanelProps {
  originalQuery?: string;
  originalResponse: string;
  analysisResult: AnalysisResponse;
}

export function ClosedLoopCorrectionPanel({
  originalQuery,
  originalResponse,
  analysisResult,
}: ClosedLoopCorrectionPanelProps) {
  const [isExecuting, setIsExecuting] = useState(false);
  const [currentStep, setCurrentStep] = useState<"idle" | "generating" | "reverifying" | "complete">("idle");
  const [correctionData, setCorrectionData] = useState<CorrectionResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const isVerifiedAlready = analysisResult.risk_level === "VERIFIED" || analysisResult.overall_h_score < 0.20;

  const handleRunCorrection = async () => {
    setIsExecuting(true);
    setCurrentStep("generating");
    try {
      // Step simulation for visual explainability progression
      setTimeout(() => {
        setCurrentStep("reverifying");
      }, 700);

      const res = await correctResponse({
        query: originalQuery?.trim() || undefined,
        response: originalResponse.trim(),
        trace_id: analysisResult.trace_id,
        model_name: "default",
      });

      setCorrectionData(res);
      setCurrentStep("complete");

      if (res.correction.status === "verified") {
        toast.success("Evidence-grounded correction successfully generated and re-verified!");
      } else if (res.correction.status === "abstained") {
        toast.info("System safely abstained from correction due to missing reference evidence.");
      } else if (res.correction.status === "rejected") {
        toast.error("Proposed candidate was rejected by the independent verification gate.");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to execute correction.";
      toast.error(msg);
      setCurrentStep("idle");
    } finally {
      setIsExecuting(false);
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success("Corrected output copied to clipboard.");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Card id="closed-loop-panel" className="border border-white/10 bg-[#090d16] overflow-hidden shadow-xl">
      {/* Header */}
      <CardHeader className="border-b border-white/5 bg-white/[0.02]">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <CardTitle className="text-base font-bold text-white">
                Explainable Closed-Loop: DETECT → EXPLAIN → CORRECT → RE-VERIFY
              </CardTitle>
            </div>
            <p className="text-xs text-[var(--text-muted)]">
              Evidence-directed correction and independent re-verification gate.
            </p>
          </div>

          {/* Step Badges */}
          <div className="flex items-center gap-1.5 text-[10px] font-mono">
            <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">
              1. DETECT
            </span>
            <ArrowRight className="w-3 h-3 text-slate-600" />
            <span className="px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 font-semibold">
              2. EXPLAIN
            </span>
            <ArrowRight className="w-3 h-3 text-slate-600" />
            <span className={cn(
              "px-2 py-0.5 rounded border font-semibold",
              currentStep === "generating" ? "bg-amber-500/20 text-amber-300 border-amber-500/40 animate-pulse" : "bg-purple-500/10 text-purple-400 border-purple-500/20"
            )}>
              3. CORRECT
            </span>
            <ArrowRight className="w-3 h-3 text-slate-600" />
            <span className={cn(
              "px-2 py-0.5 rounded border font-semibold",
              currentStep === "reverifying" ? "bg-teal-500/20 text-teal-300 border-teal-500/40 animate-pulse" : "bg-teal-500/10 text-teal-400 border-teal-500/20"
            )}>
              4. RE-VERIFY
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-5 space-y-5">
        {/* If original claim is already verified */}
        {isVerifiedAlready ? (
          <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/20 flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-semibold text-emerald-300">All Claims Grounded in Evidence</h4>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                The response has passed external evidence retrieval (FE = {(analysisResult.pillar_scores?.retrieval ?? 0.0).toFixed(2)}) and shows no factual contradiction. No correction or repair is required.
              </p>
            </div>
          </div>
        ) : (
          <>
            {/* Step 1 & 2: Diagnosis Attribution Callout */}
            <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/5 space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <Info className="w-4 h-4 text-sky-400" />
                <span>Explainable Diagnostic Synthesis</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed font-sans">
                {(analysisResult.pillar_scores?.retrieval ?? 0.0) > 0.50
                  ? "Pillar 1 detected external factual error or contradiction. Although internal model uncertainty (CG) or generation consistency (CF) may appear stable, external reference evidence establishes that the factual claim is invalid."
                  : "The claim requires closer inspection due to inconclusive evidence or elevated uncertainty."}
              </p>
            </div>

            {/* Action Button (if not yet executed) */}
            {currentStep === "idle" && (
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl bg-gradient-to-r from-emerald-950/20 via-sky-950/20 to-purple-950/20 border border-white/10">
                <div>
                  <h4 className="text-sm font-bold text-white">Generate Evidence-Grounded Correction</h4>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Extracts verified facts from retrieved evidence passages or symbolic engine, produces a candidate repair, and independently re-verifies it.
                  </p>
                </div>
                <Button
                  onClick={handleRunCorrection}
                  disabled={isExecuting}
                  size="default"
                  className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium shrink-0 cursor-pointer shadow-lg shadow-emerald-900/30"
                >
                  <Sparkles className="w-4 h-4 mr-2" />
                  Execute Closed-Loop Repair
                </Button>
              </div>
            )}

            {/* Active Execution Loading Indicator */}
            {(currentStep === "generating" || currentStep === "reverifying") && (
              <div className="p-6 rounded-xl bg-white/[0.02] border border-white/10 text-center space-y-3">
                <Loader2 className="w-6 h-6 text-emerald-400 animate-spin mx-auto" />
                <div>
                  <p className="text-sm font-semibold text-white">
                    {currentStep === "generating" ? "Generating Evidence-Grounded Candidate..." : "Executing Independent Re-Verification Gate..."}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    {currentStep === "generating" ? "Extracting entailed facts and resolving contradictory propositions" : "Sending proposed candidate through the full 3-pillar verification pipeline"}
                  </p>
                </div>
              </div>
            )}

            {/* Step 3 & 4: Correction Results Display */}
            <AnimatePresence>
              {correctionData && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-4"
                >
                  {/* Case A: Successfully Verified Correction */}
                  {correctionData.correction.status === "verified" && (
                    <div className="space-y-4">
                      {/* Prominent Corrected Output Box */}
                      <div className="p-5 rounded-xl bg-gradient-to-b from-emerald-950/40 to-[#0c1420] border-2 border-emerald-500/40 shadow-2xl space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 text-[11px] font-bold uppercase tracking-wider font-mono">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              ✓ CORRECTION VERIFIED
                            </span>
                            <span className="text-xs text-slate-400 font-mono">
                              Method: {correctionData.correction.method.replace(/_/g, " ")}
                            </span>
                          </div>

                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleCopy(correctionData.correction.corrected_text || "")}
                              className="text-xs text-slate-300 hover:text-white"
                            >
                              {copied ? <Check className="w-3.5 h-3.5 mr-1 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 mr-1" />}
                              {copied ? "Copied" : "Copy Output"}
                            </Button>
                          </div>
                        </div>

                        {/* Corrected Text */}
                        <div className="p-4 rounded-lg bg-black/40 border border-emerald-500/20">
                          <p className="text-xs font-semibold uppercase tracking-wider text-emerald-400 font-mono mb-1">
                            Final Corrected Response
                          </p>
                          <p className="text-base sm:text-lg font-medium text-white leading-relaxed font-sans">
                            {correctionData.correction.corrected_text}
                          </p>
                        </div>

                        {/* Side-by-Side Comparison */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 text-xs">
                          <div className="p-3 rounded-lg bg-red-950/20 border border-red-500/20">
                            <span className="text-[10px] font-mono uppercase text-red-400 font-bold block mb-1">
                              Original (Flagged)
                            </span>
                            <p className="text-slate-300 line-through opacity-80">{correctionData.correction.original_text}</p>
                          </div>
                          <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-500/20">
                            <span className="text-[10px] font-mono uppercase text-emerald-400 font-bold block mb-1">
                              Evidence Grounded
                            </span>
                            <p className="text-white font-medium">{correctionData.correction.corrected_text}</p>
                          </div>
                        </div>

                        {/* Supporting Evidence Citations */}
                        {correctionData.correction.supporting_evidence.length > 0 && (
                          <div className="pt-2">
                            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5 font-mono">
                              <BookOpen className="w-3.5 h-3.5 text-sky-400" />
                              Supporting Verified Evidence
                            </p>
                            <div className="space-y-1.5">
                              {correctionData.correction.supporting_evidence.map((ev, i) => (
                                <div key={i} className="p-2.5 rounded-lg bg-white/[0.02] border border-white/5 text-xs text-slate-300">
                                  <span className="font-semibold text-emerald-400 font-mono mr-2">[{ev.source}]:</span>
                                  {ev.snippet}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Re-Verification Audit Block */}
                        {correctionData.correction.reverification && (
                          <div className="mt-3 p-3 rounded-lg bg-white/[0.02] border border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                            <div className="flex items-center gap-3">
                              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                              <span className="text-slate-400">
                                Recalculated Re-Verification Score: <strong className="text-emerald-400 font-mono">H = {(correctionData.correction.reverification.overall_h_score * 100).toFixed(1)}%</strong> (P1 FE: {(correctionData.correction.reverification.pillar_scores.evidence_grounding * 100).toFixed(1)}%)
                              </span>
                            </div>
                            <Link href={`/traces?id=${correctionData.correction.reverification.trace_id}`}>
                              <Badge variant="outline" size="sm" className="cursor-pointer hover:border-emerald-500/40 text-emerald-400">
                                <GitBranch className="w-3 h-3 mr-1" /> View Re-Verification Trace
                              </Badge>
                            </Link>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Case B: Safely Abstained */}
                  {correctionData.correction.status === "abstained" && (
                    <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-3">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-amber-400" />
                        <h4 className="text-sm font-bold text-amber-300 uppercase tracking-wider font-mono">
                          Correction Safely Abstained
                        </h4>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {correctionData.correction.reason}
                      </p>
                      {correctionData.correction.missing_evidence_explanation && (
                        <div className="p-3 rounded-lg bg-black/30 border border-amber-500/20 text-xs text-slate-400">
                          <strong className="text-amber-400">Missing Evidence: </strong>
                          {correctionData.correction.missing_evidence_explanation}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Case C: Rejected by Re-verification Gate */}
                  {correctionData.correction.status === "rejected" && (
                    <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 space-y-3">
                      <div className="flex items-center gap-2">
                        <XCircle className="w-5 h-5 text-red-400" />
                        <h4 className="text-sm font-bold text-red-300 uppercase tracking-wider font-mono">
                          Correction Candidate Rejected
                        </h4>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {correctionData.correction.reason}
                      </p>
                      <p className="text-[11px] text-slate-500">
                        The original response remains flagged as unverified. HalluciSense never accepts unverified candidates.
                      </p>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}
      </CardContent>
    </Card>
  );
}
