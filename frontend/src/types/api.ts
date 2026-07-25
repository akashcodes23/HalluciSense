// ─────────────────────────────────────────────────────────────────────────────
// Core API types
// ─────────────────────────────────────────────────────────────────────────────
export interface APIError {
  detail: string;
  status_code?: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Auth
// ─────────────────────────────────────────────────────────────────────────────
export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  full_name: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface UserResponse {
  id: string;
  email: string;
  full_name: string;
  avatar_url: string | null;
  role: 'USER' | 'ADMIN';
  preferred_model: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chats
// ─────────────────────────────────────────────────────────────────────────────
export interface Chat {
  id: string;
  title: string;
  model_used: string;
  is_archived: boolean;
  last_message_at: string | null;
  created_at: string;
  message_count?: number;
}

export interface ChatListResponse {
  items: Chat[];
  total: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Messages
// ─────────────────────────────────────────────────────────────────────────────
export type MessageRole = 'USER' | 'ASSISTANT' | 'SYSTEM';
export type VerificationStatus = 'PENDING' | 'PROCESSING' | 'COMPLETE' | 'FAILED';

export interface Message {
  id: string;
  chat_id: string;
  role: MessageRole;
  content: string;
  verification_status: VerificationStatus;
  processing_time_ms: number | null;
  created_at: string;
  verification_report?: VerificationReport;
}

// ─────────────────────────────────────────────────────────────────────────────
// Verification
// ─────────────────────────────────────────────────────────────────────────────
export type RiskLevel = 'VERIFIED' | 'NEEDS_VERIFICATION' | 'LIKELY_HALLUCINATED';

export interface EvidenceItem {
  id: string;
  claim: string;
  snippet: string;
  source_name: string;
  source_url: string;
  similarity_score: number;
  is_supporting: boolean;
}

export interface SentenceAnalysis {
  id: string;
  sentence_index: number;
  sentence_text: string;
  h_score: number;
  risk_level: RiskLevel;
  color_code: string;
  factual_error: number;
  confidence_gap: number;
  consistency_failure: number;
  evidence: EvidenceItem[];
}

export interface VerificationReport {
  id: string;
  message_id: string;
  overall_h_score: number;
  overall_risk_level: RiskLevel;
  factual_error_score: number;
  confidence_gap_score: number;
  consistency_failure_score: number;
  weights_used: Record<string, number>;
  processing_time_ms: number;
  sentence_analyses: SentenceAnalysis[];
  created_at: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming
// ─────────────────────────────────────────────────────────────────────────────
export interface StreamChunk {
  type: 'token' | 'verification_dispatched' | 'error' | 'done';
  content?: string;
  message_id?: string;
  error?: string;
}
