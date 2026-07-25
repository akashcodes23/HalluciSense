import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHr = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHr / 24);

  if (diffSec < 60) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString();
}

export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength) + '...';
}

export function getRiskColor(riskLevel: string): string {
  switch (riskLevel) {
    case 'VERIFIED': return '#22c55e';
    case 'NEEDS_VERIFICATION': return '#f59e0b';
    case 'LIKELY_HALLUCINATED': return '#ef4444';
    default: return '#6b7280';
  }
}

export function getRiskBg(riskLevel: string): string {
  switch (riskLevel) {
    case 'VERIFIED': return 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30';
    case 'NEEDS_VERIFICATION': return 'bg-amber-500/15 text-amber-400 border-amber-500/30';
    case 'LIKELY_HALLUCINATED': return 'bg-red-500/15 text-red-400 border-red-500/30';
    default: return 'bg-gray-500/15 text-gray-400 border-gray-500/30';
  }
}

export function formatHScore(score: number): string {
  return (score * 100).toFixed(1) + '%';
}
