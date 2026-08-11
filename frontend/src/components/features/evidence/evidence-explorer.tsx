"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  BookOpen,
  CheckCircle2,
  AlertTriangle,
  Check,
  Copy,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
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
  supporting = [],
  contradicting = [],
}: EvidenceExplorerProps) {
  const [filter, setFilter] = useState("");

  const allItems = [...evidence, ...(explainEvidence || [])].filter(
    (item, index, self) =>
      index === self.findIndex((t) => (t.id && t.id === item.id) || t.snippet === item.snippet)
  );

  const filtered = allItems.filter(
    (item) =>
      (item.title || "").toLowerCase().includes(filter.toLowerCase()) ||
      (item.snippet || "").toLowerCase().includes(filter.toLowerCase()) ||
      (item.source || item.source_name || "").toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Search Bar */}
      {allItems.length > 3 && (
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <Input
            placeholder="Search evidence passages..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="pl-9 bg-white/[0.03] border-white/[0.08]"
          />
        </div>
      )}

      {/* Highlights: Supporting & Contradicting Passages */}
      {(supporting.length > 0 || contradicting.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {supporting.length > 0 && (
            <div className="p-4 rounded-xl border border-emerald-500/20 bg-emerald-500/[0.03] space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400">
                <CheckCircle2 className="w-4 h-4" />
                Supporting Passages ({supporting.length})
              </div>
              <ul className="space-y-1.5 text-xs text-slate-300">
                {supporting.map((s, i) => (
                  <li key={i} className="pl-3 border-l-2 border-emerald-500/40">
                    &quot;{s}&quot;
                  </li>
                ))}
              </ul>
            </div>
          )}

          {contradicting.length > 0 && (
            <div className="p-4 rounded-xl border border-rose-500/20 bg-rose-500/[0.03] space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-rose-400">
                <AlertTriangle className="w-4 h-4" />
                Contradicting Passages ({contradicting.length})
              </div>
              <ul className="space-y-1.5 text-xs text-slate-300">
                {contradicting.map((c, i) => (
                  <li key={i} className="pl-3 border-l-2 border-rose-500/40">
                    &quot;{c}&quot;
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Main Evidence Cards */}
      {filtered.length === 0 ? (
        <div className="text-center py-12 text-slate-500 text-sm">
          {allItems.length === 0
            ? "No evidence items available for this analysis."
            : "No evidence items match your search filter."}
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((item, index) => (
            <EvidenceCard key={item.id || index} item={item} index={index} />
          ))}
        </div>
      )}
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

  const scoreVal = item.score ?? item.similarity_score ?? 0;
  const similarityPct = (scoreVal * 100).toFixed(0);
  const isHighScore = scoreVal >= 0.8;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.3 }}
      className="rounded-xl border border-white/[0.06] bg-white/[0.02] overflow-hidden hover:border-white/[0.1] transition-all duration-200"
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left cursor-pointer"
        aria-expanded={isExpanded}
      >
        <BookOpen className="w-4 h-4 text-blue-400 shrink-0" />
        <span className="text-sm font-medium text-slate-200 truncate flex-1">
          {item.title || item.claim || item.source_name || "Reference Passage"}
        </span>
        <Badge
          variant={isHighScore ? "verified" : "info"}
          className="shrink-0 text-[10px]"
        >
          {similarityPct}% match
        </Badge>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-slate-500 shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-500 shrink-0" />
        )}
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="px-4 pb-4 border-t border-white/[0.04] bg-white/[0.01]"
          >
            <p className="text-sm text-slate-300 mt-3 leading-relaxed font-serif italic">
              &quot;{item.snippet}&quot;
            </p>

            <div className="flex items-center justify-between mt-3 pt-2 border-t border-white/[0.04] text-xs text-slate-500">
              <span className="font-mono">
                Source: {item.source || item.source_name || "Wikipedia Knowledge Base"}
              </span>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1 hover:text-white transition-colors cursor-pointer"
                >
                  {copied ? (
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
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
