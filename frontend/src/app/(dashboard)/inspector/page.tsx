'use client';

import React, { useState, useMemo } from 'react';
import {
  Cpu,
  Layers,
  Search,
  Copy,
  Check,
  Code,
  Sparkles,
  Zap,
  Sliders,
  SlidersHorizontal,
} from 'lucide-react';
import { toast } from 'sonner';

interface FeatureItem {
  index: number;
  name: string;
  group: 'Pillar 1 (Evidence)' | 'Pillar 2 (Consistency)' | 'Meta Signals & Fusion';
  desc: string;
  formula?: string;
}

const FEATURE_SCHEMA: FeatureItem[] = [
  { index: 0, name: 'p1_mean_entailment', group: 'Pillar 1 (Evidence)', desc: 'Average evidence entailment score across claims', formula: '\\frac{1}{N} \\sum_{i=1}^N \\text{Entailment}(c_i, E_i)' },
  { index: 1, name: 'p1_max_entailment', group: 'Pillar 1 (Evidence)', desc: 'Maximum single claim entailment score', formula: '\\max_i \\text{Entailment}(c_i, E_i)' },
  { index: 2, name: 'p1_mean_contradiction', group: 'Pillar 1 (Evidence)', desc: 'Average evidence contradiction score', formula: '\\frac{1}{N} \\sum_{i=1}^N \\text{Contradiction}(c_i, E_i)' },
  { index: 3, name: 'p1_min_support_margin', group: 'Pillar 1 (Evidence)', desc: 'Minimum support margin (entailment - contradiction)', formula: '\\min_i (\\text{Entailment}_i - \\text{Contradiction}_i)' },
  { index: 4, name: 'p1_num_claims', group: 'Pillar 1 (Evidence)', desc: 'Total claim count extracted from response', formula: '|\\mathcal{C}|' },
  { index: 5, name: 'p2_max_pairwise_contradiction', group: 'Pillar 2 (Consistency)', desc: 'Maximum pairwise claim self-contradiction score', formula: '\\max_{i \\neq j} \\text{Contradiction}(c_i, c_j)' },
  { index: 6, name: 'p2_mean_pairwise_contradiction', group: 'Pillar 2 (Consistency)', desc: 'Average pairwise claim self-contradiction', formula: '\\frac{2}{N(N-1)} \\sum_{i < j} \\text{Contradiction}(c_i, c_j)' },
  { index: 7, name: 'p2_max_pairwise_similarity', group: 'Pillar 2 (Consistency)', desc: 'Maximum semantic similarity between claims', formula: '\\max_{i \\neq j} \\cos(e_i, e_j)' },
  { index: 8, name: 'p2_fraction_contradictory_pairs', group: 'Pillar 2 (Consistency)', desc: 'Fraction of claim pairs flagged as contradictory', formula: '\\frac{|\\{ (i,j) : \\text{Contra}_{ij} > 0.5 \\}|}{\\binom{N}{2}}' },
  { index: 9, name: 'p2_num_claims', group: 'Pillar 2 (Consistency)', desc: 'Pillar 2 total evaluated claim count', formula: '|\\mathcal{C}_{P2}|' },
  { index: 10, name: 'prob_p1', group: 'Meta Signals & Fusion', desc: 'Base Pillar 1 logistic model hallucination probability', formula: 'P(H \\mid \\mathbf{x}_{P1})' },
  { index: 11, name: 'prob_p2', group: 'Meta Signals & Fusion', desc: 'Base Pillar 2 model hallucination probability', formula: 'P(H \\mid \\mathbf{x}_{P2})' },
  { index: 12, name: 'logit_p1', group: 'Meta Signals & Fusion', desc: 'Log-odds (logit) of Pillar 1 probability', formula: '\\ln \\frac{P_1}{1 - P_1}' },
  { index: 13, name: 'logit_p2', group: 'Meta Signals & Fusion', desc: 'Log-odds (logit) of Pillar 2 probability', formula: '\\ln \\frac{P_2}{1 - P_2}' },
  { index: 14, name: 'prob_disagreement_abs', group: 'Meta Signals & Fusion', desc: 'Absolute disagreement |P1 - P2|', formula: '|P(H \\mid P1) - P(H \\mid P2)|' },
  { index: 15, name: 'prob_mean', group: 'Meta Signals & Fusion', desc: 'Mean risk probability (P1 + P2)/2', formula: '\\frac{P_1 + P_2}{2}' },
  { index: 16, name: 'prob_max', group: 'Meta Signals & Fusion', desc: 'Maximum risk probability max(P1, P2)', formula: '\\max(P_1, P_2)' },
  { index: 17, name: 'prob_min', group: 'Meta Signals & Fusion', desc: 'Minimum risk probability min(P1, P2)', formula: '\\min(P_1, P_2)' },
  { index: 18, name: 'prob_ratio', group: 'Meta Signals & Fusion', desc: 'Probability ratio (P1 + eps) / (P2 + eps)', formula: '\\frac{P_1 + 10^{-5}}{P_2 + 10^{-5}}' },
];

export default function ModelInspectorPage() {
  const [selectedGroup, setSelectedGroup] = useState<string>('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [copied, setCopied] = useState(false);

  const groups = ['All', 'Pillar 1 (Evidence)', 'Pillar 2 (Consistency)', 'Meta Signals & Fusion'];

  const filteredFeatures = useMemo(() => {
    return FEATURE_SCHEMA.filter((f) => {
      const matchesGroup = selectedGroup === 'All' || f.group === selectedGroup;
      const matchesSearch =
        searchQuery === '' ||
        f.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        f.desc.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesGroup && matchesSearch;
    });
  }, [selectedGroup, searchQuery]);

  const copyPythonSchema = () => {
    const code = `# HalluciSense 19-Dimensional Hybrid Feature Vector Schema (Phase 6M)
FEATURE_SCHEMA = [
${FEATURE_SCHEMA.map((f) => `    {"index": ${f.index}, "name": "${f.name}", "group": "${f.group}"},`).join('\n')}
]
DECISION_THRESHOLD = 0.54
SCALER = "RobustScaler(quantile_range=(25.0, 75.0))"
CLASSIFIER = "HistGradientBoostingClassifier(max_iter=100, max_depth=4)"`;

    navigator.clipboard.writeText(code);
    setCopied(true);
    toast.success('Python feature schema copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex-1 h-full overflow-y-auto p-6 md:p-8 bg-bg text-slate-100 space-y-8">
      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-white/[0.06] pb-6">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-mono">
              Architecture Inspector
            </span>
            <span className="text-xs text-slate-400 font-mono">Phase 6M Hybrid Fusion Engine</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight mt-2 text-white">
            Model & Feature Inspector
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-2xl">
            Detailed inspection of the 19-dimensional hybrid feature schema, model registry parameters, and scaler transforms.
          </p>
        </div>

        <button
          onClick={copyPythonSchema}
          className="px-3.5 py-2 rounded-xl border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.06] text-slate-300 hover:text-white transition-colors flex items-center gap-2 text-xs font-mono cursor-pointer shrink-0"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Code className="w-3.5 h-3.5 text-indigo-400" />}
          <span>{copied ? 'Copied' : 'Export Python Schema'}</span>
        </button>
      </div>

      {/* ── Model Health Architecture Cards ───────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-bg-surface border border-white/[0.06] rounded-2xl p-5 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Classifier</span>
            <Zap className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-base font-bold text-emerald-400">HistGradientBoosting</div>
          <div className="text-[11px] text-slate-400 font-mono">max_iter=100, max_depth=4</div>
        </div>

        <div className="bg-bg-surface border border-white/[0.06] rounded-2xl p-5 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Preprocessor</span>
            <Sliders className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <div className="text-base font-bold text-indigo-400">RobustScaler</div>
          <div className="text-[11px] text-slate-400 font-mono">IQR Median & Quantile Scaling</div>
        </div>

        <div className="bg-bg-surface border border-white/[0.06] rounded-2xl p-5 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Decision Boundary</span>
            <SlidersHorizontal className="w-3.5 h-3.5 text-purple-400" />
          </div>
          <div className="text-base font-bold text-purple-400 font-mono">&tau;* = 0.540</div>
          <div className="text-[11px] text-slate-400 font-mono">Optimized for F1 / MCC</div>
        </div>

        <div className="bg-bg-surface border border-white/[0.06] rounded-2xl p-5 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Feature Vector</span>
            <Layers className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div className="text-base font-bold text-amber-400 font-mono">19 Dimensions</div>
          <div className="text-[11px] text-slate-400 font-mono">P1 (5) + P2 (5) + Meta (9)</div>
        </div>
      </div>

      {/* ── Feature Vector Table Card ──────────────────────────────── */}
      <div className="bg-bg-surface border border-white/[0.06] rounded-2xl p-6 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              <span>19-Dimensional Hybrid Feature Schema</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Showing {filteredFeatures.length} of {FEATURE_SCHEMA.length} feature dimensions
            </p>
          </div>

          {/* Search Box */}
          <div className="relative min-w-[240px]">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search feature or formula..."
              className="w-full bg-white/[0.02] border border-white/[0.06] rounded-xl pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-indigo-500/50 font-mono"
            />
          </div>
        </div>

        {/* Group Tabs */}
        <div className="flex items-center gap-2 flex-wrap border-b border-white/[0.04] pb-3">
          {groups.map((grp) => (
            <button
              key={grp}
              onClick={() => setSelectedGroup(grp)}
              className={`px-3 py-1 text-xs rounded-lg font-mono transition-colors cursor-pointer ${
                selectedGroup === grp
                  ? 'bg-indigo-600 text-white font-semibold shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 bg-white/[0.02] hover:bg-white/[0.04]'
              }`}
            >
              {grp}
            </button>
          ))}
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className="border-b border-white/[0.06] text-slate-400 text-xs uppercase font-mono tracking-wider">
                <th className="py-3 px-4 w-16">Index</th>
                <th className="py-3 px-4">Feature Name</th>
                <th className="py-3 px-4">Family</th>
                <th className="py-3 px-4">Description</th>
                <th className="py-3 px-4 text-right">Mathematical Representation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
              {filteredFeatures.map((feat) => (
                <tr key={feat.name} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3.5 px-4 font-mono text-xs text-slate-400">[{feat.index.toString().padStart(2, '0')}]</td>
                  <td className="py-3.5 px-4 font-mono font-medium text-indigo-300">{feat.name}</td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold ${
                        feat.group.includes('Pillar 1')
                          ? 'bg-blue-500/10 text-blue-300 border border-blue-500/20'
                          : feat.group.includes('Pillar 2')
                          ? 'bg-purple-500/10 text-purple-300 border border-purple-500/20'
                          : 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20'
                      }`}
                    >
                      {feat.group}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-xs text-slate-300">{feat.desc}</td>
                  <td className="py-3.5 px-4 text-right font-mono text-xs text-slate-400">
                    <code className="bg-white/[0.04] px-2 py-0.5 rounded border border-white/[0.06] text-slate-300">
                      ${feat.formula}$
                    </code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
