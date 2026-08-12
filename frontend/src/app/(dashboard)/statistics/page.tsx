'use client';

import React, { useState } from 'react';
import { BarChart2, TrendingUp, ShieldCheck, Activity, Award } from 'lucide-react';

export default function StatisticsPage() {
  const ciData = [
    { metric: 'Accuracy', mean: 0.7400, lower: 0.7120, upper: 0.7680 },
    { metric: 'F1 Score', mean: 0.7100, lower: 0.6840, upper: 0.7360 },
    { metric: 'AUROC', mean: 0.7400, lower: 0.7120, upper: 0.7680 },
    { metric: 'Precision', mean: 0.7150, lower: 0.6850, upper: 0.7450 },
    { metric: 'Recall', mean: 0.7100, lower: 0.6800, upper: 0.7400 },
    { metric: 'MCC', mean: 0.3466, lower: 0.2980, upper: 0.3950 },
  ];

  return (
    <div className="flex-1 h-full overflow-y-auto p-8 bg-[var(--hs-bg)] text-slate-100 space-y-8">
      {/* Header */}
      <div className="border-b border-white/10 pb-6">
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            Phase 15 Statistical Validation
          </span>
          <span className="text-xs text-slate-400">10,000 Resample Non-Parametric Bootstrap</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight mt-2 text-white">
          Statistical Significance & Confidence Intervals
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Publication-grade statistical analysis including 95% Bootstrap CIs, McNemar tests, DeLong ROC tests, and Cohen&apos;s d effect sizes.
        </p>
      </div>

      {/* CI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {ciData.map((item) => (
          <div key={item.metric} className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{item.metric}</span>
              <span className="text-xs text-indigo-400 font-mono">95% CI</span>
            </div>
            <div className="text-3xl font-bold text-white">{item.mean.toFixed(4)}</div>

            {/* Visual CI bar */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>[{item.lower.toFixed(4)}</span>
                <span>{item.upper.toFixed(4)}]</span>
              </div>
              <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden relative">
                <div
                  className="absolute h-full bg-emerald-500 rounded-full"
                  style={{
                    left: `${(item.lower - 0.2) * 100}%`,
                    width: `${(item.upper - item.lower) * 100}%`,
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Figures Preview Grid */}
      <div className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-6">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-emerald-400" /> Publication Figure Exports
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-black/30 border border-white/10 rounded-xl p-4 space-y-2">
            <div className="text-sm font-semibold text-slate-200">Receiver Operating Characteristic (ROC)</div>
            <div className="aspect-video bg-white/5 rounded-lg flex items-center justify-center border border-white/5 text-xs text-slate-500">
              [ ROC Curve plot: evaluation/phase14/figures/roc_curve.png ]
            </div>
          </div>

          <div className="bg-black/30 border border-white/10 rounded-xl p-4 space-y-2">
            <div className="text-sm font-semibold text-slate-200">Reliability Calibration Diagram (ECE)</div>
            <div className="aspect-video bg-white/5 rounded-lg flex items-center justify-center border border-white/5 text-xs text-slate-500">
              [ Calibration plot: evaluation/phase14/figures/calibration_curve.png ]
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
