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
    shortName: "Evidence",
    description: "Hybrid BM25 + Dense retrieval with NLI cross-verification",
    icon: "Database",
    color: "#6366F1",
  },
  confidence: {
    name: "Confidence Estimation",
    shortName: "Confidence",
    description: "White-box logit entropy and epistemic uncertainty analysis",
    icon: "Activity",
    color: "#A855F7",
  },
  consistency: {
    name: "Consistency Reasoning",
    shortName: "Consistency",
    description: "Paraphrase-based self-consistency verification",
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
