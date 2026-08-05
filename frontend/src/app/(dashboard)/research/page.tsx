'use client';

import React from 'react';
import { FileText, Download, ExternalLink, Code, BookOpen } from 'lucide-react';

export default function ResearchHubPage() {
  const paperAbstract = `Large Language Models (LLMs) frequently generate plausible yet factually incorrect or self-contradictory statements, known as hallucinations. Existing detection paradigms rely either purely on external search retrieval or token-level uncertainty estimation, failing to capture complex multi-step reasoning failures. In this work, we present HalluciSense, a hybrid multi-pillar hallucination detection framework. HalluciSense integrates Evidence Consistency (Pillar 1), Structural Consistency (Pillar 2), and Token/Model Confidence Signals into a 19-dimensional hybrid feature matrix processed by a RobustScaler HistGradientBoosting meta-classifier (tau* = 0.54). Evaluated across 15 domains (N=750 claims), HalluciSense achieves an AUROC of 0.7400, F1-Score of 0.7100, and MCC of 0.3466, significantly outperforming SelfCheckGPT, RAGAS, FactScore, and standalone NLI baselines (p < 0.001, McNemar test). We provide 10,000-sample bootstrap confidence intervals, SHAP topological explainability, and an enterprise telemetry pipeline for real-time model drift monitoring.`;

  return (
    <div className="flex-1 h-full overflow-y-auto p-8 bg-[var(--hs-bg)] text-slate-100 space-y-8">
      {/* Header */}
      <div className="border-b border-white/10 pb-6">
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-500/20 text-blue-300 border border-blue-500/30">
            Phase 19 Research Artifacts
          </span>
          <span className="text-xs text-slate-400">IEEE / ACM / EMNLP Publication Package</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight mt-2 text-white">
          Research Paper & Open Source Hub
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Access the automated LaTeX source package, pre-compiled PDF research paper, citation CFF, and reproducible benchmark scripts.
        </p>
      </div>

      {/* Main Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/20 flex items-center justify-center text-indigo-400">
              <FileText className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">LaTeX Source Package</h2>
              <p className="text-xs text-slate-400">Automated paper/paper.tex generator</p>
            </div>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Includes full IEEEtran template, Section 1–8 text, embedded ROC/PR/Calibration plots, and LaTeX performance tables.
          </p>
          <div className="pt-2">
            <a
              href="/paper/paper.tex"
              download
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-colors"
            >
              <Download className="w-4 h-4" /> Download paper.tex
            </a>
          </div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center text-purple-400">
              <BookOpen className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">BibTeX Citation</h2>
              <p className="text-xs text-slate-400">Standard research citation</p>
            </div>
          </div>
          <div className="bg-black/40 border border-white/10 rounded-xl p-3 font-mono text-[11px] text-slate-300 overflow-x-auto">
            {`@article{hallucisense2026,
  title={HalluciSense: A Hybrid Multi-Pillar Hallucination Detection Framework},
  author={HalluciSense Research Team},
  journal={arXiv preprint arXiv:2608.14000},
  year={2026}
}`}
          </div>
        </div>
      </div>

      {/* Abstract Preview */}
      <div className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-3">
        <h2 className="text-lg font-semibold text-white">Abstract Preview</h2>
        <p className="text-xs text-slate-300 leading-relaxed font-sans">{paperAbstract}</p>
      </div>
    </div>
  );
}
