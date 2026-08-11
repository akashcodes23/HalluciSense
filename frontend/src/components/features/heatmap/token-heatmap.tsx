"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { TOKEN_TIERS } from "@/lib/constants";
import type { TokenHeatmapItem } from "@/types/hallucisense";

interface TokenHeatmapProps {
  tokens: TokenHeatmapItem[];
}

export function TokenHeatmap({ tokens }: TokenHeatmapProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  if (!tokens || !tokens.length) {
    return (
      <div className="text-center py-12 text-slate-500 text-sm">
        No token-level heatmap data available.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Legend */}
      <div className="flex items-center gap-4 flex-wrap">
        {Object.entries(TOKEN_TIERS).map(([tier, info]) => (
          <div key={tier} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: info.bg, border: `1px solid ${info.color}30` }}
            />
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">{info.label}</span>
          </div>
        ))}
      </div>

      {/* Token Grid */}
      <div className="rounded-2xl border border-white/[0.06] bg-[#0b1220] p-6">
        <div className="flex flex-wrap gap-1 leading-relaxed relative">
          {tokens.map((token, i) => {
            const tierKey = (token.tier as keyof typeof TOKEN_TIERS) || (token.is_hallucination_suspect ? "RED" : "GREEN");
            const tier = TOKEN_TIERS[tierKey] || TOKEN_TIERS.GREEN;
            const isHovered = hoveredIndex === i;
            const probPct = token.probability != null ? (token.probability * 100).toFixed(1) : (100 - ((token.score || 0) * 100)).toFixed(1);

            return (
              <span
                key={i}
                className={cn(
                  "relative inline-block px-1 py-0.5 rounded-md cursor-pointer transition-all duration-150",
                  "text-xs font-mono",
                  isHovered && "ring-1 ring-white/30 scale-105 z-10"
                )}
                style={{
                  backgroundColor: tier.bg,
                  color: tier.color,
                  borderBottom: `2px solid ${tier.color}50`,
                }}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
              >
                {token.token}

                {/* Hover Tooltip */}
                <AnimatePresence>
                  {isHovered && (
                    <motion.div
                      initial={{ opacity: 0, y: 4, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 4, scale: 0.95 }}
                      transition={{ duration: 0.12 }}
                      className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 p-2 rounded-lg bg-[#111827] border border-white/10 shadow-xl text-left pointer-events-none z-30 min-w-[120px]"
                    >
                      <div className="text-[10px] font-mono text-slate-400">Token</div>
                      <div className="text-xs font-mono font-bold text-white tracking-tight truncate">&quot;{token.token}&quot;</div>
                      <div className="mt-1 flex items-center justify-between text-[10px] text-slate-400">
                        <span>Confidence:</span>
                        <span className="font-mono text-white">{probPct}%</span>
                      </div>
                      {token.entropy != null && (
                        <div className="flex items-center justify-between text-[10px] text-slate-400">
                          <span>Entropy:</span>
                          <span className="font-mono text-amber-400">{token.entropy.toFixed(2)}</span>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
