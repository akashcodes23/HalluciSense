// ─────────────────────────────────────────────────────────────────────────────
// HalluciSense v1.0 — Formatting Utilities
// ─────────────────────────────────────────────────────────────────────────────

export function formatLatency(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function formatScore(score: number): string {
  return (score * 100).toFixed(1);
}

export function formatPercentage(value: number): string {
  return `${value.toFixed(1)}%`;
}

export function formatMemory(mb: number): string {
  if (mb < 1024) return `${mb.toFixed(0)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
}

export function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

export function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function getRiskColor(level: string): string {
  switch (level) {
    case "VERIFIED":
      return "#22C55E";
    case "LOW_RISK":
      return "#38BDF8";
    case "MODERATE_RISK":
    case "NEEDS_VERIFICATION":
      return "#F59E0B";
    case "LIKELY_HALLUCINATED":
      return "#EF4444";
    default:
      return "#94A3B8";
  }
}

export function getRiskLabel(level: string): string {
  switch (level) {
    case "VERIFIED":
      return "Verified";
    case "LOW_RISK":
      return "Low Risk";
    case "MODERATE_RISK":
      return "Moderate Risk";
    case "NEEDS_VERIFICATION":
      return "Needs Verification";
    case "LIKELY_HALLUCINATED":
      return "Likely Hallucinated";
    default:
      return level;
  }
}
