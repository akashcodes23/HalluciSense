'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ShieldCheck, ShieldAlert, ShieldX, Info, ChevronDown, ChevronUp, CheckCircle2, AlertTriangle, XCircle, Activity, BarChart2 } from 'lucide-react';
import { useUIStore } from '../../stores/uiStore';
import { PillarCard } from './PillarCard';
import { EvidenceCard } from './EvidenceCard';

const riskConfig: Record<string, { icon: React.ReactNode; label: string; color: string; bg: string; border: string }> = {
  VERIFIED: {
    icon: <ShieldCheck className="w-5 h-5 text-emerald-400" />,
    label: 'Verified',
    color: '#10b981',
    bg: 'rgba(16,185,129,0.12)',
    border: 'rgba(16,185,129,0.3)',
  },
  NEEDS_VERIFICATION: {
    icon: <ShieldAlert className="w-5 h-5 text-amber-400" />,
    label: 'Needs Review',
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.12)',
    border: 'rgba(245,158,11,0.3)',
  },
  NEEDS_VERIFY: {
    icon: <ShieldAlert className="w-5 h-5 text-amber-400" />,
    label: 'Needs Review',
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.12)',
    border: 'rgba(245,158,11,0.3)',
  },
  MODERATE_RISK: {
    icon: <AlertTriangle className="w-5 h-5 text-orange-400" />,
    label: 'Moderate Risk',
    color: '#f97316',
    bg: 'rgba(249,115,22,0.12)',
    border: 'rgba(249,115,22,0.3)',
  },
  LIKELY_HALLUCINATED: {
    icon: <ShieldX className="w-5 h-5 text-red-400" />,
    label: 'Likely Hallucinated',
    color: '#ef4444',
    bg: 'rgba(239,68,68,0.12)',
    border: 'rgba(239,68,68,0.3)',
  },
};

function CircularGauge({ score }: { score: number }) {
  const isAvailable = typeof score === 'number' && !isNaN(score);
  const normalizedScore = isAvailable ? Math.max(0, Math.min(1, score)) : 0;
  const percent = Math.round(normalizedScore * 100);
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percent / 100) * circumference;

  let gaugeColor = '#10b981';
  if (percent > 25) gaugeColor = '#f59e0b';
  if (percent > 50) gaugeColor = '#f97316';
  if (percent > 75) gaugeColor = '#ef4444';

  return (
    <div className="relative w-24 h-24 flex items-center justify-center shrink-0">
      <svg className="w-full h-full transform -rotate-90">
        <circle
          cx="48"
          cy="48"
          r={radius}
          stroke="currentColor"
          strokeWidth="6"
          className="text-white/10"
          fill="transparent"
        />
        <motion.circle
          cx="48"
          cy="48"
          r={radius}
          stroke={gaugeColor}
          strokeWidth="6"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: isAvailable ? strokeDashoffset : circumference }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          strokeLinecap="round"
          fill="transparent"
        />
      </svg>
      <div className="absolute text-center">
        <span className="text-xl font-bold font-mono text-white">{isAvailable ? `${percent}%` : 'N/A'}</span>
        <span className="block text-[9px] uppercase tracking-wider text-slate-400 font-semibold font-mono">H-Score</span>
      </div>
    </div>
  );
}

function getPillarColor(score: number): 'green' | 'yellow' | 'red' {
  if (typeof score !== 'number' || isNaN(score) || score < 0.35) return 'green';
  if (score < 0.65) return 'yellow';
  return 'red';
}

export function VerificationPanel() {
  const { isPanelOpen, panelWidth, setPanelWidth, activeReport, activeSentenceIndex, closePanel, setActiveSentence } = useUIStore();
  const [expandedIndex, setExpandedIndex] = useState<number | null>(activeSentenceIndex ?? 0);

  // Close on Escape key
  React.useEffect(() => {
    if (!isPanelOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        closePanel();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPanelOpen, closePanel]);

  if (!isPanelOpen || !activeReport) return null;

  const sentences = activeReport.sentence_analyses || [];
  const overallConfig = riskConfig[activeReport.overall_risk] || riskConfig.VERIFIED;

  const verifiedCount = sentences.filter((s) => s.risk_level === 'VERIFIED').length;
  const reviewCount = sentences.filter((s) => s.risk_level === 'NEEDS_VERIFICATION').length;
  const hallucinatedCount = sentences.filter((s) => s.risk_level === 'LIKELY_HALLUCINATED').length;

  return (
    <AnimatePresence>
      <motion.aside
        initial={{ x: '100%', opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: '100%', opacity: 0 }}
        transition={{ type: 'spring', damping: 30, stiffness: 350, mass: 0.8 }}
        style={{ width: panelWidth }}
        className="fixed right-0 top-0 bottom-0 bg-[var(--bg-2)] border-l border-white/5 flex flex-col z-50 shadow-2xl"
      >
        {/* ── Drag Handle ── */}
        <div
          className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-teal-500/30 transition-all duration-200 ease-out z-50 opacity-0 hover:opacity-100"
          onPointerDown={(e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startWidth = panelWidth;
            const onPointerMove = (moveEvent: PointerEvent) => {
              const deltaX = startX - moveEvent.clientX;
              setPanelWidth(Math.max(320, Math.min(startWidth + deltaX, 850)));
            };
            const onPointerUp = () => {
              document.removeEventListener('pointermove', onPointerMove);
              document.removeEventListener('pointerup', onPointerUp);
            };
            document.addEventListener('pointermove', onPointerMove);
            document.addEventListener('pointerup', onPointerUp);
          }}
        />

        {/* ── Panel Header ── */}
        <div className="flex items-center justify-between p-6 border-b border-white/5">
          <div className="flex items-center gap-2">
            {overallConfig.icon}
            <div>
              <h2 className="text-sm font-bold text-white">Hallucination Inspector</h2>
              <p className="text-xs text-slate-400">Message Verification Audit Report</p>
            </div>
          </div>
          <button
            onClick={closePanel}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors text-slate-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* ── Scrollable Body ── */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* ── Enterprise Overview Card ── */}
          <div
            className="rounded-2xl p-5 border flex items-center justify-between gap-4"
            style={{ background: overallConfig.bg, borderColor: overallConfig.border }}
          >
            <div className="space-y-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Overall Verdict</span>
              <h3 className="text-lg font-bold" style={{ color: overallConfig.color }}>
                {overallConfig.label}
              </h3>
              <p className="text-xs text-slate-300">
                Confidence: {Math.round((1.0 - activeReport.overall_h_score) * 100)}%
              </p>
            </div>
            <CircularGauge score={activeReport.overall_h_score} />
          </div>

          {/* ── Evidence Statistics Grid ── */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
              <BarChart2 className="w-3.5 h-3.5 text-teal-400" /> Evidence Metrics
            </h3>
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-white/[0.03] border border-white/5 p-3 rounded-xl text-center">
                <span className="text-xs text-slate-400 block">Verified</span>
                <span className="text-lg font-bold text-green-400">{verifiedCount}</span>
              </div>
              <div className="bg-white/[0.03] border border-white/5 p-3 rounded-xl text-center">
                <span className="text-xs text-slate-400 block">Review</span>
                <span className="text-lg font-bold text-yellow-400">{reviewCount}</span>
              </div>
              <div className="bg-white/[0.03] border border-white/5 p-3 rounded-xl text-center">
                <span className="text-xs text-slate-400 block">Hallucinated</span>
                <span className="text-lg font-bold text-red-400">{hallucinatedCount}</span>
              </div>
            </div>
          </div>

          {/* ── Sentence Accordion Inspector ── */}
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-teal-400" /> Claim Breakdown ({sentences.length})
            </h3>
            <div className="space-y-3">
              {sentences.map((sentence, idx) => {
                const isExpanded = expandedIndex === idx;
                const cfg = riskConfig[sentence.risk_level] || riskConfig.VERIFIED;

                return (
                  <div
                    key={idx}
                    className="bg-white/[0.02] border border-white/5 rounded-2xl overflow-hidden transition-all duration-200"
                  >
                    {/* Sentence Accordion Header */}
                    <button
                      onClick={() => {
                        setExpandedIndex(isExpanded ? null : idx);
                        setActiveSentence(idx);
                      }}
                      className="w-full p-4 text-left flex items-start justify-between gap-3 hover:bg-white/[0.03] transition-colors"
                    >
                      <div className="flex items-start gap-2.5 flex-1 min-w-0">
                        <span
                          className="w-2 h-2 rounded-full mt-1.5 shrink-0"
                          style={{ backgroundColor: cfg.color }}
                        />
                        <span className="text-xs text-slate-200 leading-relaxed line-clamp-2">
                          &quot;{sentence.sentence_text}&quot;
                        </span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <span
                          className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-full"
                          style={{ background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}` }}
                        >
                          {cfg.label}
                        </span>
                        <ChevronDown
                          className={`w-4 h-4 text-slate-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                        />
                      </div>
                    </button>

                    {/* Sentence Details */}
                    {isExpanded && (
                      <div className="p-4 border-t border-white/5 space-y-4 bg-white/[0.01]">
                        {/* Sentence Pillars */}
                        <div className="grid grid-cols-1 gap-2.5">
                          <PillarCard
                            label="P1 · Evidence Support"
                            score={sentence.factual_error}
                            description="External evidence retrieval and NLI entailment evaluation."
                            color={getPillarColor(sentence.factual_error)}
                          />
                          <PillarCard
                            label="P2 · Model Uncertainty"
                            score={sentence.confidence_gap}
                            description="Token log-probability entropy and generation uncertainty."
                            color={getPillarColor(sentence.confidence_gap)}
                          />
                          <PillarCard
                            label="P3 · Generation Consistency"
                            score={sentence.consistency_failure}
                            description="Semantic agreement across stochastic generations."
                            color={getPillarColor(sentence.consistency_failure)}
                          />
                        </div>

                        {/* NLI Engine Reasoning */}
                        {sentence.reasoning && (
                          <div className="bg-white/[0.02] border border-white/5 p-3 rounded-xl flex gap-2.5">
                            <Info className="w-4 h-4 text-teal-400 shrink-0 mt-0.5" />
                            <p className="text-xs text-slate-300 leading-relaxed">{sentence.reasoning}</p>
                          </div>
                        )}

                        {/* Sentence Evidence Items */}
                        {sentence.evidence && sentence.evidence.length > 0 && (
                          <div className="space-y-2">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block">
                              Matched Evidence ({sentence.evidence.length})
                            </span>
                            {sentence.evidence.map((ev, evIdx) => (
                              <EvidenceCard key={evIdx} evidence={ev} />
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </motion.aside>
    </AnimatePresence>
  );
}
