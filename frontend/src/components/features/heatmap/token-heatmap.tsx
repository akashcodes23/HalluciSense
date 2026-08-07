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

  if (!tokens.length) {
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
      <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-6">
        <div className="flex flex-wrap gap-1 leading-relaxed relative">
          {tokens.map((token, i) => {
            const tier = TOKEN_TIERS[token.tier] || TOKEN_TIERS.GREEN;
            const isHovered = hoveredIndex === i;

            return (
              <span
                key={i}
                className={cn(
                  "relative inline-block px-1 py-0.5 rounded-md cursor-pointer transition-all duration-200",
                  "text-sm font-mono",
                  isHovered && "ring-1 ring-white/20 scale-105 z-10"
                )}
                style={{
                  backgroundColor: tier.bg,
                  color: tier.color,
                  borderBottom: `2px solid ${tier.color}50`,
                }}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
                role="button"
                tabIndex={0}
                aria-label={`Token: ${token.token}, Risk: ${(token.score * 100).toFixed(0)}%, Tier: ${token.tier}`}
                onFocus={() => setHoveredIndex(i)}
                onBlur={() => setHoveredIndex(null)}
              >
                {token.token}

                {/* Tooltip */}
                <AnimatePresence>
                  {isHovered && (
                    <motion.div
                      initial={{ opacity: 0, y: 6, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 4, scale: 0.95 }}
                      transition={{ duration: 0.15 }}
                      className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50 pointer-events-none"
                    >
                      <div className="bg-[#111827] border border-white/[0.1] rounded-lg px-3 py-2 shadow-xl whitespace-nowrap">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: tier.color }} />
                            <span className="text-xs font-semibold text-white">{token.token}</span>
                          </div>
                          <div className="text-[10px] text-slate-400 space-y-0.5">
                            <div>Risk: <span className="text-slate-200 font-mono">{(token.score * 100).toFixed(1)}%</span></div>
                            <div>Tier: <span style={{ color: tier.color }}>{token.tier}</span></div>
                          </div>
                        </div>
                        {/* Arrow */}
                        <div className="absolute top-full left-1/2 -translate-x-1/2 w-2 h-2 bg-[#111827] border-r border-b border-white/[0.1] rotate-45 -mt-1" />
                      </div>
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
