'use client';

import React, { useState, useEffect } from 'react';
import { Database, Activity, Cpu, Layers, ShieldCheck, HelpCircle } from 'lucide-react';

export default function ModelInspectorPage() {
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/v1/hallucisense/health')
      .then((res) => res.json())
      .then((data) => {
        setHealth(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to fetch health status:', err);
        setLoading(false);
      });
  }, []);

  const featureSchema = [
    { name: 'p1_mean_entailment', group: 'Pillar 1', desc: 'Average evidence entailment score across claims' },
    { name: 'p1_max_entailment', group: 'Pillar 1', desc: 'Maximum single claim entailment score' },
    { name: 'p1_mean_contradiction', group: 'Pillar 1', desc: 'Average evidence contradiction score' },
    { name: 'p1_min_support_margin', group: 'Pillar 1', desc: 'Minimum support margin (entailment - contradiction)' },
    { name: 'p1_num_claims', group: 'Pillar 1', desc: 'Total claim count extracted from response' },
    { name: 'p2_max_pairwise_contradiction', group: 'Pillar 2', desc: 'Maximum pairwise claim self-contradiction score' },
    { name: 'p2_mean_pairwise_contradiction', group: 'Pillar 2', desc: 'Average pairwise claim self-contradiction' },
    { name: 'p2_max_pairwise_similarity', group: 'Pillar 2', desc: 'Maximum semantic similarity between claims' },
    { name: 'p2_fraction_contradictory_pairs', group: 'Pillar 2', desc: 'Fraction of claim pairs flagged as contradictory' },
    { name: 'p2_num_claims', group: 'Pillar 2', desc: 'Pillar 2 total evaluated claim count' },
    { name: 'prob_p1', group: 'Meta Signals', desc: 'Base Pillar 1 logistic model hallucination probability' },
    { name: 'prob_p2', group: 'Meta Signals', desc: 'Base Pillar 2 model hallucination probability' },
    { name: 'logit_p1', group: 'Meta Signals', desc: 'Log-odds (logit) of Pillar 1 probability' },
    { name: 'logit_p2', group: 'Meta Signals', desc: 'Log-odds (logit) of Pillar 2 probability' },
    { name: 'prob_disagreement_abs', group: 'Meta Signals', desc: 'Absolute disagreement |P1 - P2|' },
    { name: 'prob_mean', group: 'Meta Signals', desc: 'Mean risk probability (P1 + P2)/2' },
    { name: 'prob_max', group: 'Meta Signals', desc: 'Maximum risk probability max(P1, P2)' },
    { name: 'prob_min', group: 'Meta Signals', desc: 'Minimum risk probability min(P1, P2)' },
    { name: 'prob_ratio', group: 'Meta Signals', desc: 'Probability ratio (P1 + eps) / (P2 + eps)' },
  ];

  return (
    <div className="flex-1 h-full overflow-y-auto p-8 bg-[var(--hs-bg)] text-slate-100 space-y-8">
      {/* Header */}
      <div className="border-b border-white/10 pb-6">
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            Model Architecture Audit
          </span>
          <span className="text-xs text-slate-400">Phase 6M Hybrid Fusion Engine</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight mt-2 text-white">
          Model & Feature Inspector
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Detailed inspection of the 19-dimensional hybrid feature schema, model registry health, and preprocessing scaler parameters.
        </p>
      </div>

      {/* Model Health Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-xl p-5 space-y-1">
          <div className="text-xs text-slate-400">Active Classifier</div>
          <div className="text-lg font-bold text-emerald-400">HistGradientBoosting</div>
          <div className="text-[11px] text-slate-500">max_iter=100, max_depth=4</div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-xl p-5 space-y-1">
          <div className="text-xs text-slate-400">Preprocessor Scaler</div>
          <div className="text-lg font-bold text-indigo-400">RobustScaler</div>
          <div className="text-[11px] text-slate-500">IQR Median & Quantile Scaling</div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-xl p-5 space-y-1">
          <div className="text-xs text-slate-400">Decision Threshold</div>
          <div className="text-lg font-bold text-purple-400">τ* = 0.54</div>
          <div className="text-[11px] text-slate-500">Optimized for F1 / MCC</div>
        </div>

        <div className="bg-white/5 border border-white/10 rounded-xl p-5 space-y-1">
          <div className="text-xs text-slate-400">Feature Dimensions</div>
          <div className="text-lg font-bold text-amber-400">19 Features</div>
          <div className="text-[11px] text-slate-500">P1 (5) + P2 (5) + Meta (9)</div>
        </div>
      </div>

      {/* Feature Vector Table */}
      <div className="bg-white/5 border border-white/10 rounded-2xl p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-400" /> 19-Dimensional Hybrid Feature Schema
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-white/10 text-slate-400 text-xs uppercase tracking-wider">
                <th className="py-3 px-4">Index</th>
                <th className="py-3 px-4">Feature Name</th>
                <th className="py-3 px-4">Family</th>
                <th className="py-3 px-4">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {featureSchema.map((feat, idx) => (
                <tr key={feat.name} className="hover:bg-white/5 transition-colors">
                  <td className="py-3 px-4 font-mono text-xs text-slate-500">[{idx.toString().padStart(2, '0')}]</td>
                  <td className="py-3 px-4 font-mono font-medium text-indigo-300">{feat.name}</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                      feat.group === 'Pillar 1' ? 'bg-blue-500/20 text-blue-300' :
                      feat.group === 'Pillar 2' ? 'bg-purple-500/20 text-purple-300' :
                      'bg-emerald-500/20 text-emerald-300'
                    }`}>
                      {feat.group}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-xs text-slate-400">{feat.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
