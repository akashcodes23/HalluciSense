/**
 * HalluciSense Verification Pipeline API Contracts.
 * Synchronized with FastAPI Backend Production Schemas (`app.schemas.production_schemas`).
 */

export type RiskLevel = 'VERIFIED' | 'NEEDS_VERIFICATION' | 'MODERATE_RISK' | 'LIKELY_HALLUCINATED';

export type AttributionDirection = 'hallucination_risk' | 'protective' | 'neutral';

export interface LocalAttributionFeature {
  feature_name: string;
  index: number;
  value: number;
  baseline: number;
  attribution: number;
  direction: AttributionDirection;
}

export interface LocalAttribution {
  method: 'local_counterfactual_attribution';
  feature_count: number;
  baseline_type: 'training_median_from_robust_scaler';
  original_probability: number;
  baseline_probability: number;
  threshold: number;
  decision_margin: number;
  interaction_gap: number;
  interaction_gap_explanation: string;
  scientific_caveat: string;
  features: LocalAttributionFeature[];
  top_hallucination_drivers: LocalAttributionFeature[];
  top_protective_drivers: LocalAttributionFeature[];
  inference_count: number;
}

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
  whitebox_entropy?: number | null;
  blackbox_variation_score?: number | null;
  epistemic_uncertainty?: number | null;
  aleatoric_uncertainty?: number | null;
  methodology?: string;
  signal_type?: "MEASURED" | "DERIVED" | "PROXY" | "UNAVAILABLE";
  uncertainty_measure?: string | null;
  generations_used?: number | null;
  raw_signal_metadata?: Record<string, unknown>;
  explanation?: string | null;
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

export interface MeasuredTimingBreakdown {
  retrieval_ms?: number | null;
  bm25_ms?: number | null;
  dense_ms?: number | null;
  nli_ms?: number | null;
  gemini_generation_ms?: number | null;
  p1_latency_ms: number;
  p2_latency_ms: number;
  p3_latency_ms: number;
  fusion_latency_ms: number;
  total_latency_ms: number;
}

export interface PillarExecutionStatus {
  p1_status: string;
  p2_status: string;
  p3_status: string;
  fusion_status: string;
  p1_available: boolean;
  p2_available: boolean;
  p3_available: boolean;
  is_full_analysis: boolean;
}

export interface MathematicalFusionDecomposition {
  equation: string;
  fusion_mode: "FULL_THREE_PILLAR" | "PARTIAL_RENORMALIZED";
  configured_weights: Record<string, number>;
  effective_weights: Record<string, number>;
  pillar_scores: Record<string, number | null>;
  weighted_contributions: Record<string, number | null>;
  available_pillars?: string[];
  missing_pillars?: string[];
  uncalibrated_h_score: number;
  calibrated_h_score: number;
  is_full_analysis: boolean;
  explanation?: string | null;
}

export interface SemanticEvidenceEvaluation {
  title: string;
  url?: string;
  snippet: string;
  entailment: number;
  neutral: number;
  contradiction: number;
  label: 'entailment' | 'neutral' | 'contradiction';
  confidence: number;
}

export interface SemanticClaimGrounding {
  claim_id: number;
  claim_text: string;
  evidence_count: number;
  primary_status: string;
  max_entailment: number;
  mean_contradiction: number;
  support_margin: number;
  evidence_details: SemanticEvidenceEvaluation[];
}

export interface SemanticGrounding {
  status: string;
  shadow_only?: boolean;
  model_name?: string;
  total_claims_evaluated?: number;
  total_pairs_evaluated?: number;
  latency_ms?: number;
  aggregated_features?: {
    mean_entailment: number;
    max_entailment: number;
    mean_contradiction: number;
    min_support_margin: number;
    num_claims: number;
  };
  claims?: SemanticClaimGrounding[];
}

export interface CandidateComparison {
  candidate_model_version: string;
  shadow_only: boolean;
  candidate_probability: number;
  candidate_verdict: string;
  production_probability: number;
  production_verdict: string;
  decision_delta: number;
  verdicts_match: boolean;
}

export interface ClaimVerificationResult {
  claim_id: number;
  claim_text: string;
  claim_type: string;
  verification_method: string;
  status: 'VERIFIED' | 'CONTRADICTED' | 'INSUFFICIENT_EVIDENCE' | 'NOT_APPLICABLE' | 'ERROR';
  evidence_sufficiency: 'DIRECT_SUPPORT' | 'DIRECT_CONTRADICTION' | 'PARTIAL_SUPPORT' | 'AMBIGUOUS' | 'NO_EVIDENCE';
  confidence_band: 'HIGH' | 'MEDIUM' | 'LOW';
  verification_confidence: number;
  evidence?: SemanticEvidenceEvaluation[];
  symbolic_result?: Record<string, unknown>;
  reason?: string;
}

export interface ResponseVerificationSummary {
  request_id: string;
  trace_id: string;
  total_claims: number;
  verified_claims: number;
  contradicted_claims: number;
  unsupported_claims: number;
  error_claims: number;
  primary_status: string;
  model_score: number;
  model_threshold: number;
  is_hallucinated: boolean;
  claims: ClaimVerificationResult[];
}

export interface AnalysisResponse {
  request_id?: string;
  trace_id?: string;
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
  latency_ms?: number;
  processing_time_ms?: number;
  version?: string;
  measured_timings?: MeasuredTimingBreakdown;
  pillar_status?: PillarExecutionStatus;
  fusion_decomposition?: MathematicalFusionDecomposition;
  local_attribution?: LocalAttribution;
  semantic_grounding?: SemanticGrounding;
  candidate_comparison?: CandidateComparison;
  verification_summary?: ResponseVerificationSummary;
}

export interface AnalysisHistoryEntry {
  id: string;
  query: string;
  response: string;
  result: AnalysisResponse;
  timestamp: string;
}

export interface ExplainRequest {
  query?: string;
  response?: string;
  model_name?: string;
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
  fusion_decomposition?: MathematicalFusionDecomposition;
  measured_timings?: MeasuredTimingBreakdown;
}

export interface MetricsResponse {
  requests: number;
  success_rate: number | null;
  error_rate: number | null;
  avg_h_score?: number | null;
  average_h_score?: number | null;
  average_latency_ms: number | null;
  active_models?: number;
  total_requests?: number;
  verifications_completed?: number;
  avg_confidence?: number | null;
  avg_latency_ms?: number | null;
  memory_mb?: number;
  status?: string;
  risk_distribution?: Record<string, number>;
  pillar_averages?: Record<string, number>;
  throughput_rpm?: number;
  uptime_seconds?: number;
}

export interface TraceStage {
  name: string;
  status: "completed" | "running" | "failed" | "pending" | "success" | "unavailable" | "skipped";
  duration_ms: number | null;
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
  performance_timings?: Record<string, unknown>;
  measured_timings?: MeasuredTimingBreakdown;
  pillar_status?: PillarExecutionStatus;
  fusion_decomposition?: MathematicalFusionDecomposition;
}

export type VerificationState = 'IDLE' | 'CONNECTING' | 'ANALYZING' | 'COMPLETED' | 'FAILED';
