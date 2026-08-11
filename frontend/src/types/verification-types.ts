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
  claim: string;
  snippet: string;
  source_name: string;
  source_url?: string;
  similarity_score?: number;
  retrieval_method?: string;
  publish_year?: number;
}

export interface SentenceScore {
  sentence_index: number;
  sentence_text: string;
  h_score: number;
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
  pillar1_factual_error: number;
  pillar2_confidence_gap?: number | null;
  pillar3_consistency_failure?: number | null;
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
}

export interface TokenHeatmapItem {
  token: string;
  probability: number;
  entropy: number;
  is_hallucination_suspect: boolean;
}

export interface AnalysisRequest {
  text: string;
  query?: string;
  provided_evidence?: EvidenceItem[];
  model_name?: string;
  temperature?: number;
  top_p?: number;
  logprobs?: number[];
  sample_responses?: string[];
}

export interface AnalysisResponse {
  overall_h_score: number;
  risk_level: RiskLevel;
  flagged_sentences_count: number;
  total_sentences_count: number;
  pillar_scores: PillarScores;
  sentence_scores: SentenceScore[];
  confidence_analysis?: ConfidenceAnalysis;
  token_heatmap?: TokenHeatmapItem[];
  root_cause_classification?: string;
  trace_id?: string;
  latency_ms?: number;
}

export interface ExplainRequest {
  analysis_response: AnalysisResponse;
  original_query?: string;
}

export interface ExplainResponse {
  explanation_markdown: string;
  remediation_suggestions: string[];
  key_evidence_citations: string[];
}

export type VerificationState = 'IDLE' | 'CONNECTING' | 'ANALYZING' | 'COMPLETED' | 'FAILED';
