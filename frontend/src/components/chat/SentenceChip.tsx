'use client';

import React from 'react';
import { useUIStore } from '../../stores/uiStore';

interface SentenceChipProps {
  text: string;
  sentenceIndex: number;
  riskLevel: 'VERIFIED' | 'NEEDS_VERIFICATION' | 'LIKELY_HALLUCINATED';
  hScore: number;
  messageId: string;
  report: any;
}

const riskStyles = {
  VERIFIED: {
    bg: 'transparent',
    border: 'rgba(34, 197, 94, 0.5)',
    hover: 'rgba(34, 197, 94, 0.05)',
    dot: '#22c55e',
  },
  NEEDS_VERIFICATION: {
    bg: 'transparent',
    border: 'rgba(245, 158, 11, 0.5)',
    hover: 'rgba(245, 158, 11, 0.05)',
    dot: '#f59e0b',
  },
  LIKELY_HALLUCINATED: {
    bg: 'transparent',
    border: 'rgba(239, 68, 68, 0.5)',
    hover: 'rgba(239, 68, 68, 0.05)',
    dot: '#ef4444',
  },
};

export function SentenceChip({ text, sentenceIndex, riskLevel, hScore, messageId, report }: SentenceChipProps) {
  const { openPanel } = useUIStore();
  const styles = riskStyles[riskLevel];

  const handleClick = () => {
    openPanel(messageId, report, sentenceIndex);
  };

  return (
    <span
      onClick={handleClick}
      title={`H-Score: ${(hScore * 100).toFixed(0)}% — Click to inspect`}
      style={{
        backgroundColor: styles.bg,
        borderBottom: `2px solid ${styles.border}`,
        transition: 'background-color 0.2s ease, border-color 0.2s ease',
        cursor: 'pointer',
        display: 'inline',
        paddingBottom: '2px',
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.backgroundColor = styles.hover;
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.backgroundColor = styles.bg;
      }}
    >
      {text}
      <span
        style={{
          display: 'inline-block',
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: styles.dot,
          marginLeft: '3px',
          marginBottom: '2px',
          verticalAlign: 'middle',
          flexShrink: 0,
        }}
      />
    </span>
  );
}
