"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Loader2, RotateCcw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { useAnalysis, useExplain } from "@/hooks/use-analysis";
import { useAnalysisStore } from "@/store/analysis-store";
import { MODEL_OPTIONS } from "@/lib/constants";
import { PipelineAnimation } from "@/components/features/analyzer/pipeline-animation";
import { ResultDashboard } from "@/components/features/analyzer/result-dashboard";
import { toast } from "sonner";

export default function AnalyzePage() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [model, setModel] = useState("GPT-4");

  const analysis = useAnalysis();
  const explain = useExplain();
  const currentResult = useAnalysisStore((s) => s.currentResult);
  const currentExplain = useAnalysisStore((s) => s.currentExplain);
  const reset = useAnalysisStore((s) => s.reset);

  const isLoading = analysis.isPending || explain.isPending;

  const handleAnalyze = async () => {
    if (!query.trim() || !response.trim()) {
      toast.error("Please enter both a question and an LLM response.");
      return;
    }

    reset();

    const payload = { query: query.trim(), response: response.trim(), model_name: model };

    try {
      await analysis.mutateAsync(payload);
      // Fire explain in parallel for detailed results
      explain.mutate(payload);
      toast.success("Analysis complete");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Analysis failed";
      toast.error(message);
    }
  };

  const handleReset = () => {
    setQuery("");
    setResponse("");
    reset();
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 shadow-[0_0_24px_rgba(37,99,235,0.25)]">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">Analyzer</h1>
              <p className="text-sm text-slate-500">Three-pillar hallucination detection</p>
            </div>
          </div>
        </motion.div>

        {/* ── Input Form ─────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="space-y-5"
        >
          {/* Query */}
          <div className="space-y-2">
            <label className="label" htmlFor="query-input">Question / Context</label>
            <Textarea
              id="query-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What question was asked to the LLM?"
              className="min-h-[100px]"
              disabled={isLoading}
            />
          </div>

          {/* Response */}
          <div className="space-y-2">
            <label className="label" htmlFor="response-input">LLM Response</label>
            <Textarea
              id="response-input"
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              placeholder="Paste the LLM-generated response to verify..."
              className="min-h-[160px]"
              disabled={isLoading}
            />
          </div>

          {/* Controls Row */}
          <div className="flex items-center gap-3 flex-wrap">
            {/* Model Selector */}
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isLoading}
              className="h-10 px-3 rounded-xl border border-white/[0.08] bg-white/[0.03] text-sm text-slate-300 focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 cursor-pointer disabled:opacity-50"
              aria-label="Select model"
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.value} value={m.value} className="bg-[#111827]">
                  {m.label}
                </option>
              ))}
            </select>

            <div className="flex-1" />

            {/* Reset */}
            {(query || response || currentResult) && (
              <Button variant="ghost" size="sm" onClick={handleReset} disabled={isLoading}>
                <RotateCcw className="w-4 h-4" />
                Reset
              </Button>
            )}

            {/* Analyze Button */}
            <Button
              onClick={handleAnalyze}
              disabled={isLoading || !query.trim() || !response.trim()}
              size="lg"
              className="min-w-[160px]"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Analyze
                </>
              )}
            </Button>
          </div>
        </motion.div>

        {/* ── Pipeline Animation ──────────────────────────────────────── */}
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <PipelineAnimation isActive={isLoading} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Results ─────────────────────────────────────────────────── */}
        <AnimatePresence>
          {currentResult && !isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            >
              <ResultDashboard
                result={currentResult}
                explain={currentExplain}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Empty State ─────────────────────────────────────────────── */}
        {!currentResult && !isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="flex flex-col items-center justify-center py-16 text-center"
          >
            <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-4">
              <Zap className="w-7 h-7 text-slate-600" />
            </div>
            <p className="text-slate-500 text-sm max-w-sm">
              Enter a question and the LLM response above, then click{" "}
              <span className="text-blue-400 font-medium">Analyze</span> to detect hallucinations
              using the three-pillar pipeline.
            </p>
          </motion.div>
        )}
      </div>
    </div>
  );
}
