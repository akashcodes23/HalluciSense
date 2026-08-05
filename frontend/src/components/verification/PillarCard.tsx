'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface PillarCardProps {
  label: string;
  score: number;
  description: string;
  color: 'green' | 'yellow' | 'red';
}

const colorMap = {
  green:  { bar: '#22c55e', bg: 'rgba(34,197,94,0.12)',  border: 'rgba(34,197,94,0.25)',  text: '#4ade80' },
  yellow: { bar: '#f59e0b', bg: 'rgba(245,158,11,0.12)', border: 'rgba(245,158,11,0.25)', text: '#fbbf24' },
  red:    { bar: '#ef4444', bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.25)',  text: '#f87171' },
};

function getPillarColor(score: number): 'green' | 'yellow' | 'red' {
  if (score < 0.35) return 'green';
  if (score < 0.65) return 'yellow';
  return 'red';
}

export function PillarCard({ label, score, description, color }: PillarCardProps) {
  const isAvailable = typeof score === 'number' && !isNaN(score);
  const normalizedScore = isAvailable ? Math.max(0, Math.min(1, score)) : 0;
  const pct = Math.round(normalizedScore * 100);
  const c = colorMap[color] || colorMap.green;
  const displayScore = isAvailable ? `${pct}%` : 'Unavailable';

  return (
    <div
      className="rounded-2xl p-4"
      style={{ background: c.bg, border: `1px solid ${c.border}` }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-semibold text-slate-200">{label}</span>
        <span className="text-lg font-bold" style={{ color: c.text }}>{displayScore}</span>
      </div>
      <p className="text-xs text-slate-400 mb-3 leading-relaxed">{description}</p>

      {/* Animated progress bar */}
      <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: c.bar }}
          initial={{ width: 0 }}
          animate={{ width: isAvailable ? `${pct}%` : '0%' }}
          transition={{ duration: 0.7, ease: 'easeOut' }}
        />
      </div>
    </div>
  );
}
