'use client';

import React, { useState } from 'react';
import { ExternalLink, CheckCircle, XCircle, Copy, Check, Globe, BookOpen, Database, FileText } from 'lucide-react';
import { EvidenceItem } from '../../stores/uiStore';
import toast from 'react-hot-toast';

interface EvidenceCardProps {
  evidence: EvidenceItem;
}

function getSourceMeta(sourceName: string) {
  const srcLower = (sourceName || '').toLowerCase();
  if (srcLower.includes('wikipedia')) return { icon: BookOpen, color: '#3b82f6', label: 'Wikipedia' };
  if (srcLower.includes('wikidata')) return { icon: Database, color: '#9333ea', label: 'Wikidata' };
  if (srcLower.includes('pubmed')) return { icon: FileText, color: '#10b981', label: 'PubMed' };
  if (srcLower.includes('crossref')) return { icon: Globe, color: '#f59e0b', label: 'CrossRef' };
  return { icon: Globe, color: '#6366f1', label: sourceName || 'Web Source' };
}

export function EvidenceCard({ evidence }: EvidenceCardProps) {
  const [copied, setCopied] = useState(false);
  const confidence = Math.round((evidence.similarity_score || 0.8) * 100);
  const meta = getSourceMeta(evidence.source_name);
  const IconComp = meta.icon;

  const handleCopyCitation = () => {
    const citation = `"${evidence.snippet}" — ${evidence.source_name} (${evidence.source_url || 'N/A'})`;
    navigator.clipboard.writeText(citation);
    setCopied(true);
    toast.success('Citation copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white/[0.03] border border-white/10 rounded-2xl p-4 hover:border-white/20 transition-all duration-200 ease-out space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div
            className="w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold shrink-0"
            style={{ backgroundColor: `${meta.color}20`, color: meta.color }}
          >
            <IconComp className="w-3.5 h-3.5" />
          </div>
          <span className="text-xs font-semibold text-slate-200">
            {meta.label}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {evidence.is_supporting ? (
            <span className="flex items-center gap-1 text-[11px] font-semibold text-green-400 bg-green-500/10 px-2 py-0.5 rounded-full border border-green-500/20">
              <CheckCircle className="w-3 h-3" /> Supporting
            </span>
          ) : (
            <span className="flex items-center gap-1 text-[11px] font-semibold text-red-400 bg-red-500/10 px-2 py-0.5 rounded-full border border-red-500/20">
              <XCircle className="w-3 h-3" /> Contradicting
            </span>
          )}
        </div>
      </div>

      {/* Snippet */}
      <blockquote className="text-xs text-slate-300 leading-relaxed border-l-2 border-indigo-500/40 pl-3 italic">
        &quot;{evidence.snippet}&quot;
      </blockquote>

      {/* Matched Claim */}
      {evidence.claim && (
        <div className="text-[11px] text-slate-400 bg-white/[0.02] p-2 rounded-lg border border-white/5">
          <span className="font-semibold text-slate-400">Claim: </span>
          <span>{evidence.claim}</span>
        </div>
      )}

      {/* Similarity Score Bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-[10px] text-slate-400 font-mono">
          <span>Similarity Match</span>
          <span>{confidence}%</span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-white/10 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
            style={{ width: `${confidence}%`, transition: 'width 0.6s ease' }}
          />
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between pt-2 border-t border-white/5">
        <button
          onClick={handleCopyCitation}
          className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-white transition-colors"
        >
          {copied ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}
          {copied ? 'Copied' : 'Copy Citation'}
        </button>

        {evidence.source_url && (
          <a
            href={evidence.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-[11px] text-indigo-400 hover:text-indigo-300 transition-colors font-medium"
          >
            Open Source <ExternalLink className="w-3 h-3" />
          </a>
        )}
      </div>
    </div>
  );
}
