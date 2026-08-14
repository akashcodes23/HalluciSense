'use client';

import React, { useState } from 'react';
import {
  FileText,
  Download,
  BookOpen,
  Copy,
  Check,
  ExternalLink,
  Code,
  Sparkles,
  GitBranch,
} from 'lucide-react';
import { toast } from 'sonner';

export default function ResearchHubPage() {
  const [copiedBibtex, setCopiedBibtex] = useState(false);
  const [copiedAbstract, setCopiedAbstract] = useState(false);

  const paperAbstract = `Large Language Models (LLMs) frequently generate plausible yet factually incorrect or self-contradictory statements, known as hallucinations. Existing detection paradigms rely either purely on external search retrieval or token-level uncertainty estimation, failing to capture complex multi-step reasoning failures. In this work, we present HalluciSense, a hybrid multi-pillar hallucination detection framework. HalluciSense integrates Evidence Consistency (Pillar 1), Structural Consistency (Pillar 2), and Token/Model Confidence Signals into a 19-dimensional hybrid feature matrix processed by a RobustScaler HistGradientBoosting meta-classifier (tau* = 0.54). Evaluated across 15 domains (N=750 claims), HalluciSense achieves an AUROC of 0.7400, F1-Score of 0.7100, and MCC of 0.3466, significantly outperforming SelfCheckGPT, RAGAS, FactScore, and standalone NLI baselines (p < 0.001, McNemar test). We provide 10,000-sample bootstrap confidence intervals, SHAP topological explainability, and an enterprise telemetry pipeline for real-time model drift monitoring.`;

  const bibtexCitation = `@article{hallucisense2026,
  title={HalluciSense: A Hybrid Multi-Pillar Hallucination Detection Framework},
  author={Akash Patil and DeepMind Advanced Agentic Systems},
  journal={arXiv preprint arXiv:2608.14000},
  year={2026}
}`;

  const copyBibtex = () => {
    navigator.clipboard.writeText(bibtexCitation);
    setCopiedBibtex(true);
    toast.success('BibTeX citation copied to clipboard!');
    setTimeout(() => setCopiedBibtex(false), 2000);
  };

  const copyAbstract = () => {
    navigator.clipboard.writeText(paperAbstract);
    setCopiedAbstract(true);
    toast.success('Paper abstract copied to clipboard!');
    setTimeout(() => setCopiedAbstract(false), 2000);
  };

  return (
    <div className="flex-1 h-full overflow-y-auto p-6 md:p-8 bg-bg text-slate-100 space-y-8">
      {/* ── Page Header ─────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-white/[0.06] pb-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-300 border border-blue-500/20 font-mono">
              Phase 19 Research Artifacts
            </span>
            <span className="text-xs text-slate-400 font-mono">IEEE / ACM / EMNLP Publication Package</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight mt-2 text-white flex items-center gap-2">
            Research Paper & Open Source Hub
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Access the automated LaTeX source package, pre-compiled PDF research paper, citation CFF, and reproducible benchmark scripts.
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <a
            href="https://github.com/akashcodes23/HalluciSense"
            target="_blank"
            rel="noopener noreferrer"
            className="px-3.5 py-2 rounded-xl border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.06] text-slate-300 hover:text-white transition-colors flex items-center gap-2 text-xs font-mono"
          >
            <GitBranch className="w-3.5 h-3.5 text-purple-400" />
            <span>GitHub Repository</span>
            <ExternalLink className="w-3 h-3 text-slate-400" />
          </a>
        </div>
      </div>

      {/* ── Action Cards Grid ───────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* LaTeX Package Card */}
        <div className="bg-bg-surface border border-white/[0.06] rounded-2xl p-6 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                <FileText className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-bold text-white">LaTeX Manuscript Source</h2>
                <p className="text-xs text-slate-400 font-mono">paper/paper.tex generator</p>
              </div>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Includes full IEEEtran template, Section 1–8 text, embedded ROC/PR/Calibration plots, and multi-pillar LaTeX comparison tables.
            </p>
          </div>

          <div className="pt-2">
            <a
              href="/paper/paper.tex"
              download
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-colors cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download paper.tex</span>
            </a>
          </div>
        </div>

        {/* BibTeX Citation Card */}
        <div className="bg-bg-surface border border-white/[0.06] rounded-2xl p-6 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
                  <BookOpen className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-white">BibTeX Citation</h2>
                  <p className="text-xs text-slate-400 font-mono">Standard research citation</p>
                </div>
              </div>

              <button
                onClick={copyBibtex}
                className="px-2.5 py-1 text-xs rounded-lg border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.06] text-slate-300 hover:text-white transition-colors flex items-center gap-1.5 font-mono cursor-pointer"
              >
                {copiedBibtex ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3 text-slate-400" />}
                <span>{copiedBibtex ? 'Copied' : 'Copy'}</span>
              </button>
            </div>

            <div className="bg-black/40 border border-white/[0.06] rounded-xl p-3 font-mono text-[11px] text-slate-300 overflow-x-auto leading-relaxed">
              {bibtexCitation}
            </div>
          </div>
        </div>
      </div>

      {/* ── Abstract Card ───────────────────────────────────────────── */}
      <div className="bg-bg-surface border border-white/[0.06] rounded-2xl p-6 space-y-3">
        <div className="flex items-center justify-between border-b border-white/[0.04] pb-3">
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-accent-primary" />
            <span>Abstract Preview</span>
          </h2>

          <button
            onClick={copyAbstract}
            className="px-2.5 py-1 text-xs rounded-lg border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.06] text-slate-300 hover:text-white transition-colors flex items-center gap-1.5 font-mono cursor-pointer"
          >
            {copiedAbstract ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3 text-slate-400" />}
            <span>{copiedAbstract ? 'Copied' : 'Copy Abstract'}</span>
          </button>
        </div>

        <p className="text-xs text-slate-300 leading-relaxed font-sans">{paperAbstract}</p>
      </div>
    </div>
  );
}
