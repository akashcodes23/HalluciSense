// ─────────────────────────────────────────────────────────────────────────────
// HalluciSense v1.0 — Production API TypeScript Types
// Synchronized with FastAPI Backend Schemas
// ─────────────────────────────────────────────────────────────────────────────

export type RiskLevel =
  | "VERIFIED"
  | "NEEDS_VERIFICATION"
  | "MODERATE_RISK"
  | "LIKELY_HALLUCINATED"
  | "LOW_RISK";

export interface EvidenceItem {
  id?: string;
  title?: string;
  claim?: string;
  snippet: string;
  score?: number;
  similarity_score?: number;
  source?: string;
  source_name?: string;
  source_url?: string;
  publish_year?: number;
}

export interface SentenceScore {
  sentence_index: number;
  text?: string;
  sentence_text?: string;
  score?: number;
  h_score?: number;
  risk_level: RiskLevel;
  epistemic_category?: string;
  nli_entailment_prob?: number;
  nli_contradiction_prob?: number;
  nli_neutral_prob?: number;
  evidence_matched?: EvidenceItem[];
  temporal_anchor?: {
    asserted_year?: number;
    evidence_year?: number;
    is_compatible?: boolean;
  };
  reasoning_summary?: string;
}

export interface PillarScores {
  pillar1_factual_error?: number;
  pillar2_confidence_gap?: number | null;
  pillar3_consistency_failure?: number | null;
  retrieval?: number;
  confidence?: number;
  consistency?: number;
  effective_weights?: {
    alpha: number;
    beta: number;
    gamma: number;
  };
}

export interface TokenHeatmapItem {
  token: string;
  score?: number;
  probability?: number;
  entropy?: number;
  tier?: "GREEN" | "YELLOW" | "ORANGE" | "RED";
  color_hex?: string;
  is_hallucination_suspect?: boolean;
}

export interface ConfidenceAnalysis {
  token_entropy_mean?: number;
  confidence_gap_score?: number;
  uncertainty_level?: string;
  whitebox_entropy?: number;
  blackbox_variation_score?: number;
  epistemic_uncertainty?: number;
  aleatoric_uncertainty?: number;
}

export interface AnalysisRequest {
  text?: string;
  response?: string;
  query?: string;
  provided_evidence?: EvidenceItem[];
  model_name?: string;
  temperature?: number;
  top_p?: number;
  logprobs?: number[];
  sample_responses?: string[];
}

export interface AnalysisResponse {
  trace_id?: string;
  overall_h_score: number;
  risk_level: RiskLevel;
  confidence?: number;
  flagged_sentences_count?: number;
  total_sentences_count?: number;
  pillar_scores: PillarScores;
  sentence_scores: SentenceScore[];
  token_heatmap?: TokenHeatmapItem[];
  evidence?: EvidenceItem[];
  confidence_analysis?: ConfidenceAnalysis | null;
  root_cause_classification?: string | null;
  processing_time_ms?: number;
  latency_ms?: number;
  version?: string;
  hallucination?: boolean;
  failure_taxonomy?: string;
}

export interface ExplainRequest {
  analysis_response?: AnalysisResponse;
  query?: string;
  response?: string;
  model_name?: string;
  original_query?: string;
}

export interface ExplainResponse {
  trace_id?: string;
  overall_h_score?: number;
  risk_level?: string;
  explanation_markdown?: string;
  remediation_suggestions?: string[];
  key_evidence_citations?: string[];
  retrieved_evidence?: EvidenceItem[];
  supporting_passages?: string[];
  contradiction_evidence?: string[];
  token_heatmap?: TokenHeatmapItem[];
  sentence_scores?: SentenceScore[];
  reasoning_chain?: string[];
  fusion_contribution?: Record<string, number>;
  adaptive_weights?: Record<string, number>;
  confidence_explanation?: string;
}

export interface MetricsResponse {
  requests: number;
  average_latency_ms: number;
  average_h_score: number;
  success_rate: number;
  error_rate: number;
  memory_mb: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds?: number;
}

export interface ReadinessResponse {
  status: "ready" | "unready";
  components?: Record<string, boolean>;
}

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

export interface AnalysisHistoryEntry {
  id: string;
  query?: string;
  response?: string;
  result: AnalysisResponse;
  explain?: ExplainResponse;
  timestamp: string;
}
