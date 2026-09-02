// ─────────────────────────────────────────────────────────────────────────────
// HalluciSense v1.0 — Design Constants
// ─────────────────────────────────────────────────────────────────────────────

export const RISK_LEVELS = {
  VERIFIED: {
    label: "Verified",
    color: "#22C55E",
    bg: "rgba(34, 197, 94, 0.1)",
    border: "rgba(34, 197, 94, 0.2)",
    icon: "ShieldCheck",
  },
  LOW_RISK: {
    label: "Low Risk",
    color: "#38BDF8",
    bg: "rgba(56, 189, 248, 0.1)",
    border: "rgba(56, 189, 248, 0.2)",
    icon: "ShieldCheck",
  },
  MODERATE_RISK: {
    label: "Moderate Risk",
    color: "#F59E0B",
    bg: "rgba(245, 158, 11, 0.1)",
    border: "rgba(245, 158, 11, 0.2)",
    icon: "ShieldAlert",
  },
  LIKELY_HALLUCINATED: {
    label: "Likely Hallucinated",
    color: "#EF4444",
    bg: "rgba(239, 68, 68, 0.1)",
    border: "rgba(239, 68, 68, 0.2)",
    icon: "ShieldX",
  },
  NEEDS_VERIFICATION: {
    label: "Needs Verification",
    color: "#F59E0B",
    bg: "rgba(245, 158, 11, 0.1)",
    border: "rgba(245, 158, 11, 0.2)",
    icon: "ShieldAlert",
  },
} as const;

export const PILLAR_INFO = {
  retrieval: {
    name: "Evidence Grounding",
    shortName: "Grounding",
    symbol: "FE",
    description: "Hybrid BM25 sparse + FAISS dense retrieval with cross-encoder NLI entailment scoring",
    icon: "Database",
    color: "#6366F1",
  },
  confidence: {
    name: "Confidence Gap",
    shortName: "Confidence",
    symbol: "CG",
    description: "Token log-probability distribution and Shannon entropy H(p) uncertainty analysis",
    icon: "Activity",
    color: "#A855F7",
  },
  consistency: {
    name: "Consistency Failure",
    shortName: "Consistency",
    symbol: "CF",
    description: "Multi-sample semantic consistency and cross-generation contradiction analysis",
    icon: "GitBranch",
    color: "#3B82F6",
  },
} as const;

export const TOKEN_TIERS = {
  GREEN: { label: "Safe", color: "#10B981", bg: "rgba(16, 185, 129, 0.15)" },
  YELLOW: { label: "Caution", color: "#F59E0B", bg: "rgba(245, 158, 11, 0.15)" },
  ORANGE: { label: "Warning", color: "#F97316", bg: "rgba(249, 115, 22, 0.15)" },
  RED: { label: "Danger", color: "#EF4444", bg: "rgba(239, 68, 68, 0.15)" },
} as const;

export const PIPELINE_STAGES = [
  { id: "retrieval", label: "Retrieving Evidence", icon: "Search" },
  { id: "nli", label: "NLI Verification", icon: "Scale" },
  { id: "confidence", label: "Confidence Analysis", icon: "Activity" },
  { id: "consistency", label: "Consistency Check", icon: "GitBranch" },
  { id: "fusion", label: "Adaptive Fusion", icon: "Layers" },
  { id: "decision", label: "Risk Assessment", icon: "ShieldCheck" },
] as const;

export const MODEL_OPTIONS = [
  { value: "GPT-4", label: "GPT-4" },
  { value: "GPT-4o", label: "GPT-4o" },
  { value: "claude", label: "Claude" },
  { value: "gemini", label: "Gemini" },
  { value: "llama-3", label: "LLaMA 3" },
  { value: "mistral", label: "Mistral" },
  { value: "deepseek", label: "DeepSeek" },
  { value: "default", label: "Default" },
] as const;

export const NAV_ITEMS = [
  { href: "/analyze", label: "Analyzer", icon: "Zap" },
  { href: "/traces", label: "Traces", icon: "GitBranch" },
  { href: "/metrics", label: "Metrics", icon: "BarChart3" },
  { href: "/settings", label: "Settings", icon: "Settings" },
] as const;

export function getPillar1Diagnostic(value?: number | null, isAvailable: boolean = true): { metricLabel: string; interpretation: string } {
  if (!isAvailable || value == null || isNaN(value)) {
    return { metricLabel: "Factual Error", interpretation: "Evidence retrieval unavailable" };
  }
  if (value <= 0.20) {
    return { metricLabel: "Factual Error", interpretation: "Strong external evidence supports this claim." };
  }
  if (value <= 0.50) {
    return { metricLabel: "Factual Error", interpretation: "Evidence is inconclusive or partially grounded." };
  }
  return { metricLabel: "Factual Error", interpretation: "Retrieved evidence contradicts or fails to support this claim." };
}

export function getPillar2Diagnostic(value?: number | null, isAvailable: boolean = true): { metricLabel: string; interpretation: string } {
  if (!isAvailable || value == null || isNaN(value)) {
    return { metricLabel: "Confidence Gap", interpretation: "Token log-probabilities not provided" };
  }
  if (value <= 0.25) {
    return { metricLabel: "Confidence Gap", interpretation: "Low confidence gap indicates low internal generation uncertainty." };
  }
  if (value <= 0.55) {
    return { metricLabel: "Confidence Gap", interpretation: "Moderate confidence gap observed across generated tokens." };
  }
  return { metricLabel: "Confidence Gap", interpretation: "Elevated confidence gap indicates greater generation uncertainty." };
}

export function getPillar3Diagnostic(value?: number | null, isAvailable: boolean = true): { metricLabel: string; interpretation: string } {
  if (!isAvailable || value == null || isNaN(value)) {
    return { metricLabel: "Consistency Failure", interpretation: "Multiple generations not available" };
  }
  if (value <= 0.20) {
    return { metricLabel: "Consistency Failure", interpretation: "Independent generations show strong semantic agreement." };
  }
  if (value <= 0.50) {
    return { metricLabel: "Consistency Failure", interpretation: "Partial semantic variation detected across generations." };
  }
  return { metricLabel: "Consistency Failure", interpretation: "Independent generations show substantial disagreement." };
}

