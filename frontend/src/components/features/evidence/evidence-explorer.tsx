"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Copy,
  Check,
  BookOpen,
  ThumbsUp,
  ThumbsDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { EvidenceItem } from "@/types/hallucisense";

interface EvidenceExplorerProps {
  evidence: EvidenceItem[];
  explainEvidence?: EvidenceItem[];
  supporting?: string[];
  contradicting?: string[];
}

export function EvidenceExplorer({
  evidence,
  explainEvidence,
  supporting,
  contradicting,
}: EvidenceExplorerProps) {
  const allEvidence = explainEvidence?.length ? explainEvidence : evidence;

  if (!allEvidence.length) {
    return (
      <div className="text-center py-12 text-slate-500 text-sm">
        No evidence retrieved for this analysis.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Supporting / Contradicting Summary */}
      {(supporting?.length || contradicting?.length) ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
          {supporting && supporting.length > 0 && (
            <div className="rounded-xl border border-emerald-500/10 bg-emerald-500/[0.04] p-4">
              <div className="flex items-center gap-2 mb-2">
                <ThumbsUp className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-xs font-medium text-emerald-400">Supporting Evidence</span>
              </div>
              <ul className="space-y-1.5">
                {supporting.slice(0, 3).map((s, i) => (
                  <li key={i} className="text-xs text-slate-400 leading-relaxed">
                    {s}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {contradicting && contradicting.length > 0 && (
            <div className="rounded-xl border border-red-500/10 bg-red-500/[0.04] p-4">
              <div className="flex items-center gap-2 mb-2">
                <ThumbsDown className="w-3.5 h-3.5 text-red-400" />
                <span className="text-xs font-medium text-red-400">Contradicting Evidence</span>
              </div>
              <ul className="space-y-1.5">
                {contradicting.slice(0, 3).map((c, i) => (
                  <li key={i} className="text-xs text-slate-400 leading-relaxed">
                    {c}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ) : null}

      {/* Evidence Cards */}
      <div className="space-y-3">
        {allEvidence.map((item, index) => (
          <EvidenceCard key={item.id || index} item={item} index={index} />
        ))}
      </div>
    </div>
  );
}

function EvidenceCard({ item, index }: { item: EvidenceItem; index: number }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(item.snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const similarityPct = (item.score * 100).toFixed(0);
  const isHighScore = item.score >= 0.8;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.3 }}
      className="rounded-xl border border-white/[0.06] bg-white/[0.02] overflow-hidden hover:border-white/[0.1] transition-all duration-200"
    >
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left cursor-pointer"
        aria-expanded={isExpanded}
      >
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-white/[0.04] shrink-0">
          <BookOpen className="w-4 h-4 text-slate-500" />
        </div>

        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-slate-200 truncate">{item.title}</p>
          <p className="text-xs text-slate-500 truncate">{item.source}</p>
        </div>

        <Badge variant={isHighScore ? "verified" : "default"}>
          {similarityPct}%
        </Badge>

        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-slate-500 shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-500 shrink-0" />
        )}
      </button>

      {/* Expanded Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1 border-t border-white/[0.04]">
              <p className="text-sm text-slate-300 leading-relaxed mt-3 mb-3">
                {item.snippet}
              </p>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.06] text-xs text-slate-400 hover:text-white hover:bg-white/[0.08] transition-all cursor-pointer"
                >
                  {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
