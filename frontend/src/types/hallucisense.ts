// ─────────────────────────────────────────────────────────────────────────────
// HalluciSense v1.0 — Production API TypeScript Types
// Matches backend/app/schemas/production_schemas.py exactly
// ─────────────────────────────────────────────────────────────────────────────

// ── Request Types ────────────────────────────────────────────────────────────

export interface AnalysisRequest {
  query: string;
  response: string;
  model_name?: string;
}

export interface ExplainRequest {
  query: string;
  response: string;
  model_name?: string;
}

// ── Pillar Scores ────────────────────────────────────────────────────────────

export interface PillarScores {
  retrieval: number;
  confidence: number;
  consistency: number;
}

// ── Sentence-Level Score ─────────────────────────────────────────────────────

export interface SentenceScore {
  sentence_index: number;
  text: string;
  score: number;
  risk_level: string;
}

// ── Token Heatmap ────────────────────────────────────────────────────────────

export interface TokenHeatmapItem {
  token: string;
  score: number;
  tier: "GREEN" | "YELLOW" | "ORANGE" | "RED";
  color_hex: string;
}

// ── Evidence Item ────────────────────────────────────────────────────────────

export interface EvidenceItem {
  id: string;
  title: string;
  snippet: string;
  score: number;
  source: string;
}

// ── Confidence Analysis ──────────────────────────────────────────────────────

export interface ConfidenceAnalysis {
  whitebox_entropy: number;
  blackbox_variation_score: number;
  epistemic_uncertainty: number;
  aleatoric_uncertainty: number;
}

// ── Analysis Response ────────────────────────────────────────────────────────

export type RiskLevel =
  | "VERIFIED"
  | "LOW_RISK"
  | "MODERATE_RISK"
  | "LIKELY_HALLUCINATED";

export interface AnalysisResponse {
  trace_id: string;
  overall_h_score: number;
  risk_level: RiskLevel;
  confidence: number;
  pillar_scores: PillarScores;
  failure_taxonomy: string;
  processing_time_ms: number;
  version: string;
  hallucination: boolean;
  sentence_scores: SentenceScore[];
  token_heatmap: TokenHeatmapItem[];
  evidence: EvidenceItem[];
  confidence_analysis: ConfidenceAnalysis | null;
  root_cause_classification: string | null;
}

// ── Explain Response ─────────────────────────────────────────────────────────

export interface ExplainResponse {
  trace_id: string;
  overall_h_score: number;
  risk_level: string;
  retrieved_evidence: EvidenceItem[];
  supporting_passages: string[];
  contradiction_evidence: string[];
  token_heatmap: TokenHeatmapItem[];
  sentence_scores: SentenceScore[];
  reasoning_chain: string[];
  fusion_contribution: Record<string, number>;
  adaptive_weights: Record<string, number>;
  confidence_explanation: string;
}

// ── Metrics Response ─────────────────────────────────────────────────────────

export interface MetricsResponse {
  requests: number;
  average_latency_ms: number;
  average_h_score: number;
  success_rate: number;
  error_rate: number;
  memory_mb: number;
}

// ── Health & Readiness ───────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds?: number;
}

export interface ReadinessResponse {
  status: "ready" | "unready";
  components?: Record<string, boolean>;
}

// ── Debug Trace ──────────────────────────────────────────────────────────────

export interface TraceStage {
  name: string;
  duration_ms: number;
  status: string;
  output?: Record<string, unknown>;
}

export interface TraceSummary {
  final_h_score: number;
  risk_level: string;
  root_cause_classification: string;
}

export interface TraceData {
  trace_id: string;
  timestamp: string;
  stages: TraceStage[];
  summary: TraceSummary;
  raw?: Record<string, unknown>;
}

// ── Analysis History Entry ───────────────────────────────────────────────────

export interface AnalysisHistoryEntry {
  id: string;
  query: string;
  response: string;
  result: AnalysisResponse;
  explain?: ExplainResponse;
  timestamp: string;
}
