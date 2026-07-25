'use client';

import React from 'react';
import { ExternalLink, CheckCircle, XCircle } from 'lucide-react';
import { EvidenceItem } from '../../stores/uiStore';

interface EvidenceCardProps {
  evidence: EvidenceItem;
}

export function EvidenceCard({ evidence }: EvidenceCardProps) {
  const confidence = Math.round(evidence.similarity_score * 100);

  return (
    <div className="glass rounded-2xl p-4 border border-white/5 hover:border-white/10 transition-all duration-200 ease-out">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          {evidence.is_supporting ? (
            <CheckCircle className="w-4 h-4 text-green-400 shrink-0" />
          ) : (
            <XCircle className="w-4 h-4 text-red-400 shrink-0" />
          )}
          <span className={`text-xs font-semibold uppercase tracking-wider ${
            evidence.is_supporting ? 'text-green-400' : 'text-red-400'
          }`}>
            {evidence.is_supporting ? 'Supporting' : 'Contradicting'}
          </span>
        </div>
        <span className="text-xs text-slate-400 shrink-0">
          {confidence}% match
        </span>
      </div>

      {/* Snippet */}
      <blockquote className="text-sm text-slate-300 leading-relaxed border-l-2 border-white/20 pl-3 mb-3 italic">
        "{evidence.snippet}"
      </blockquote>

      {/* Source */}
      <a
        href={evidence.source_url}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
      >
        <ExternalLink className="w-3 h-3" />
        {evidence.source_name}
      </a>

      {/* Similarity bar */}
      <div className="mt-3 h-1 rounded-full bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
          style={{ width: `${confidence}%`, transition: 'width 0.6s ease' }}
        />
      </div>
    </div>
  );
}
