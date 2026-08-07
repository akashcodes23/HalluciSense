"use client";

import React from "react";
import { motion } from "framer-motion";
import { getRiskColor } from "@/lib/format";

interface ScoreGaugeProps {
  score: number;
  riskLevel: string;
  size?: number;
}

export function ScoreGauge({ score, riskLevel, size = 160 }: ScoreGaugeProps) {
  const color = getRiskColor(riskLevel);
  const radius = (size - 20) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = score * circumference;
  const center = size / 2;

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        {/* Background Ring */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.04)"
          strokeWidth={8}
        />
        {/* Score Arc */}
        <motion.circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - progress }}
          transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
          style={{
            filter: `drop-shadow(0 0 8px ${color}40)`,
          }}
        />
      </svg>

      {/* Center Label */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          className="text-3xl font-bold font-mono"
          style={{ color }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          {(score * 100).toFixed(1)}
        </motion.span>
        <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">
          H-Score
        </span>
      </div>
    </div>
  );
}
