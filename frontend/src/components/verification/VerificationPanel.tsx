'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ShieldCheck, ShieldAlert, ShieldX, Info } from 'lucide-react';
import { useUIStore, VerificationReport } from '../../stores/uiStore';
import { PillarCard } from './PillarCard';
import { EvidenceCard } from './EvidenceCard';

const riskConfig = {
  VERIFIED: {
    icon: <ShieldCheck className="w-5 h-5 text-green-400" />,
    label: 'Verified',
    color: '#22c55e',
    bg: 'rgba(34,197,94,0.12)',
    border: 'rgba(34,197,94,0.3)',
  },
  NEEDS_VERIFICATION: {
    icon: <ShieldAlert className="w-5 h-5 text-yellow-400" />,
    label: 'Needs Review',
    color: '#f59e0b',
    bg: 'rgba(245,158,11,0.12)',
    border: 'rgba(245,158,11,0.3)',
  },
  LIKELY_HALLUCINATED: {
    icon: <ShieldX className="w-5 h-5 text-red-400" />,
    label: 'Likely Hallucinated',
    color: '#ef4444',
    bg: 'rgba(239,68,68,0.12)',
    border: 'rgba(239,68,68,0.3)',
  },
};

function getPillarColor(score: number): 'green' | 'yellow' | 'red' {
  if (score < 0.35) return 'green';
  if (score < 0.65) return 'yellow';
  return 'red';
}

export function VerificationPanel() {
  const { isPanelOpen, panelWidth, setPanelWidth, activeReport, activeSentenceIndex, closePanel, setActiveSentence } = useUIStore();

  const activeSentence = activeReport?.sentence_analyses[activeSentenceIndex ?? 0];
  const overallConfig = activeReport ? riskConfig[activeReport.overall_risk] : null;
  const sentenceConfig = activeSentence ? riskConfig[activeSentence.risk_level] : null;

  return (
    <AnimatePresence>
      {isPanelOpen && activeReport && (
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
            className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-indigo-500/30 transition-all duration-200 ease-out z-50 opacity-0 hover:opacity-100"
            onPointerDown={(e) => {
              e.preventDefault();
              const startX = e.clientX;
              const startWidth = panelWidth;
              
              const onPointerMove = (moveEvent: PointerEvent) => {
                const deltaX = startX - moveEvent.clientX;
                const newWidth = Math.max(300, Math.min(startWidth + deltaX, 800));
                setPanelWidth(newWidth);
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
              {overallConfig?.icon}
              <div>
                <h2 className="text-sm font-bold text-white">Hallucination Inspector</h2>
                <p className="text-xs text-slate-400">H-Score: {(activeReport.overall_h_score * 100).toFixed(0)}%</p>
              </div>
            </div>
            <button onClick={closePanel} className="p-2 rounded-lg hover:bg-white/5 transition-colors text-slate-400 hover:text-white">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* ── Overall Risk Badge ── */}
            <div
              className="rounded-xl p-4 flex items-center gap-3"
              style={{ background: overallConfig?.bg, border: `1px solid ${overallConfig?.border}` }}
            >
              {overallConfig?.icon}
              <div>
                <p className="text-sm font-semibold" style={{ color: overallConfig?.color }}>{overallConfig?.label}</p>
                <p className="text-xs text-slate-400">Overall confidence: {(100 - activeReport.overall_h_score * 100).toFixed(0)}%</p>
              </div>
            </div>

            {/* ── Sentence Selector ── */}
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Sentences</h3>
              <div className="space-y-1">
                {activeReport.sentence_analyses.map((s, i) => {
                  const cfg = riskConfig[s.risk_level];
                  return (
                    <button
                      key={i}
                      onClick={() => setActiveSentence(i)}
                      className={`w-full text-left px-4 py-3 rounded-xl text-xs transition-all ${
                        activeSentenceIndex === i ? 'bg-white/10 text-white shadow-sm' : 'text-slate-400 hover:bg-white/5 hover:text-slate-300'
                      }`}
                    >
                      <span className="inline-block w-2 h-2 rounded-full mr-2" style={{ backgroundColor: cfg.color }} />
                      <span className="line-clamp-1">{s.sentence_text}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* ── Active Sentence Detail ── */}
            {activeSentence && (
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Selected Sentence</h3>
                <p className="text-sm text-slate-200 leading-relaxed mb-4 italic border-l-2 pl-3" style={{ borderColor: sentenceConfig?.color }}>
                  "{activeSentence.sentence_text}"
                </p>

                {/* Tri-Pillar Breakdown */}
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Tri-Pillar Analysis</h3>
                <div className="space-y-3">
                  <PillarCard
                    label="Factual Grounding"
                    score={activeSentence.factual_error}
                    description="How well the claim is supported by retrieved evidence."
                    color={getPillarColor(activeSentence.factual_error)}
                  />
                  <PillarCard
                    label="Confidence Gap"
                    score={activeSentence.confidence_gap}
                    description="Model token probability uncertainty for this statement."
                    color={getPillarColor(activeSentence.confidence_gap)}
                  />
                  <PillarCard
                    label="Consistency Failure"
                    score={activeSentence.consistency_failure}
                    description="Semantic drift across multiple sampled responses."
                    color={getPillarColor(activeSentence.consistency_failure)}
                  />
                </div>

                {/* Engine Reasoning */}
                {activeSentence.reasoning && (
                  <div className="mt-6 bg-white/[0.02] border border-white/5 p-4 rounded-2xl flex gap-3">
                    <Info className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
                    <p className="text-[13px] text-slate-300 leading-relaxed">{activeSentence.reasoning}</p>
                  </div>
                )}

                {/* Evidence */}
                {activeSentence.evidence && activeSentence.evidence.length > 0 && (
                  <div className="mt-4">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Retrieved Evidence</h3>
                    <div className="space-y-3">
                      {activeSentence.evidence.map((ev, i) => (
                        <EvidenceCard key={i} evidence={ev} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
