/**
 * HalluciSense Verification Pipeline API Contracts.
 * Synchronized with FastAPI Backend Production Schemas (`app.schemas.production_schemas`).
 */

export type RiskLevel = 'VERIFIED' | 'NEEDS_VERIFICATION' | 'MODERATE_RISK' | 'LIKELY_HALLUCINATED';

export type EpistemicCategory = 
  | 'ASSERTED_FACT'
  | 'PREDICTION'
  | 'HYPOTHETICAL'
  | 'CONDITIONAL'
  | 'NEGATED_FACT'
  | 'QUOTED_CLAIM'
  | 'COUNTERFACTUAL'
  | 'FICTIONAL';

export interface EvidenceItem {
  id?: string;
  claim?: string;
  snippet: string;
  source?: string;
  source_name?: string;
  source_url?: string;
  similarity_score?: number;
  score?: number;
  retrieval_method?: string;
  publish_year?: number;
  title?: string;
}

export interface SentenceScore {
  sentence_index?: number;
  sentence_text?: string;
  text?: string;
  h_score?: number;
  score?: number;
  risk_level: RiskLevel;
  epistemic_category?: EpistemicCategory;
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

export interface ConfidenceAnalysis {
  token_entropy_mean?: number;
  confidence_gap_score?: number;
  uncertainty_level?: string;
  whitebox_entropy?: number;
  blackbox_variation_score?: number;
  epistemic_uncertainty?: number;
  aleatoric_uncertainty?: number;
}

export interface TokenHeatmapItem {
  token: string;
  probability?: number;
  entropy?: number;
  score?: number;
  tier?: string;
  is_hallucination_suspect?: boolean;
}

export interface AnalysisRequest {
  text?: string;
  response?: string;
  query?: string;
  provided_evidence?: (EvidenceItem | string)[];
  model_name?: string;
  temperature?: number;
  top_p?: number;
  logprobs?: number[];
  sample_responses?: string[];
}

export interface AnalysisResponse {
  overall_h_score: number;
  risk_level: RiskLevel;
  confidence?: number;
  flagged_sentences_count?: number;
  total_sentences_count?: number;
  pillar_scores: PillarScores;
  sentence_scores: SentenceScore[];
  confidence_analysis?: ConfidenceAnalysis;
  token_heatmap?: TokenHeatmapItem[];
  evidence?: EvidenceItem[];
  root_cause_classification?: string;
  failure_taxonomy?: string;
  trace_id?: string;
  latency_ms?: number;
  processing_time_ms?: number;
  version?: string;
}

export interface AnalysisHistoryEntry {
  id: string;
  query: string;
  response: string;
  result: AnalysisResponse;
  timestamp: string;
}

export interface ExplainRequest {
  analysis_response?: AnalysisResponse;
  original_query?: string;
  response?: string;
  h_score?: number;
  risk_level?: RiskLevel;
}

export interface ExplainResponse {
  explanation_markdown?: string;
  confidence_explanation?: string;
  remediation_suggestions?: string[];
  key_evidence_citations?: string[];
  retrieved_evidence?: EvidenceItem[];
  supporting_passages?: string[];
  contradiction_evidence?: string[];
  reasoning_chain?: string[];
  adaptive_weights?: Record<string, number>;
  fusion_contribution?: Record<string, number>;
}

export interface MetricsResponse {
  requests: number;
  success_rate: number;
  average_latency_ms: number;
  active_models?: number;
  total_requests?: number;
  verifications_completed?: number;
  avg_h_score?: number;
  avg_confidence?: number;
  avg_latency_ms?: number;
  risk_distribution?: Record<string, number>;
  pillar_averages?: Record<string, number>;
  throughput_rpm?: number;
  uptime_seconds?: number;
}

export interface TraceStage {
  name: string;
  status: "completed" | "running" | "failed" | "pending" | "success";
  duration_ms: number;
  memory_mb?: number;
  details?: Record<string, unknown>;
}

export interface TraceData {
  trace_id: string;
  timestamp: string;
  stages: TraceStage[];
  summary?: {
    total_duration_ms: number;
    total_memory_mb: number;
    final_h_score: number;
    risk_level: string;
    root_cause_classification: string;
    stage_count: number;
  };
}

export type VerificationState = 'IDLE' | 'CONNECTING' | 'ANALYZING' | 'COMPLETED' | 'FAILED';
