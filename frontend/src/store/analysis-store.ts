// ─────────────────────────────────────────────────────────────────────────────
// HalluciSense v1.0 — Zustand Analysis Store
// Global state for analysis results, history, and UI state
// ─────────────────────────────────────────────────────────────────────────────

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  AnalysisResponse,
  ExplainResponse,
  AnalysisHistoryEntry,
} from "@/types/hallucisense";

// ─── Error Feed Event Schema ──────────────────────────────────────────────────
export type ErrorEventSource = "VERIFY" | "CHAT" | "SYSTEM";

/**
 * VERIFIED is intentionally excluded — the Error Feed is an anomaly feed,
 * not a complete verification history.
 */
export type ErrorEventRiskLevel =
  | "LIKELY_HALLUCINATED"
  | "MODERATE_RISK"
  | "NEEDS_VERIFICATION"
  | "CORRECTED"
  | "FAILED"
  | "REVIEW";

export interface VerificationErrorEvent {
  /** Unique event ID */
  id: string;
  /** ISO-8601 timestamp */
  timestamp: string;
  /** Which flow produced this event */
  source: ErrorEventSource;
  /** Normalised risk label (VERIFIED is never stored in the feed) */
  risk_level: ErrorEventRiskLevel;
  /** User-facing query text (Verify) or user message (Chat) */
  query?: string;
  /**
   * For non-corrected events: the AI response that was analysed.
   * For CORRECTED events: the final verified response.
   * Use original_response + corrected_response for the diff panel.
   */
  response: string;
  /** H-score in [0, 1]; undefined for FAILED/SYSTEM events */
  h_score?: number;
  /** Root-cause tag from the verify pipeline */
  root_cause?: string;
  /** Failure taxonomy from the verify pipeline (if present) */
  failure_taxonomy?: string;
  /** Pillar scores from the verify pipeline */
  pillar_scores?: AnalysisResponse["pillar_scores"];
  /** Backend trace ID (if available) */
  trace_id?: string;
  /** Human-readable error description for FAILED events */
  error_message?: string;
  /** Pipeline latency in milliseconds */
  latency_ms?: number;

  // ── Correction-specific fields (CORRECTED events only) ────────────────────
  /** True when the pipeline performed an auto-correction */
  corrected?: boolean;
  /** The unsafe original draft before correction */
  original_response?: string;
  /** The verified corrected response (same as `response` for CORRECTED) */
  corrected_response?: string;
  /** Per-claim corrections from the pipeline */
  claims_corrected?: Array<Record<string, unknown>>;
}

// ─── Store Interface ──────────────────────────────────────────────────────────
interface AnalysisState {
  // Current analysis
  currentResult: AnalysisResponse | null;
  currentExplain: ExplainResponse | null;
  isAnalyzing: boolean;

  // History (persisted to localStorage)
  history: AnalysisHistoryEntry[];

  // Global error feed (persisted, max 100 entries)
  errorFeed: VerificationErrorEvent[];

  // UI state
  sidebarOpen: boolean;
  activeTab: "results" | "evidence" | "heatmap" | "trace";
  selectedTraceId: string | null;
  activeHeroState: "detect" | "confidence" | "verify";

  // Actions
  setCurrentResult: (result: AnalysisResponse | null) => void;
  setCurrentExplain: (explain: ExplainResponse | null) => void;
  setIsAnalyzing: (v: boolean) => void;
  addToHistory: (entry: AnalysisHistoryEntry) => void;
  clearHistory: () => void;
  setSidebarOpen: (open: boolean) => void;
  setActiveTab: (tab: AnalysisState["activeTab"]) => void;
  setSelectedTraceId: (id: string | null) => void;
  setActiveHeroState: (state: "detect" | "confidence" | "verify") => void;
  reset: () => void;

  // Error feed actions
  addErrorEvent: (event: VerificationErrorEvent) => void;
  removeErrorEvent: (id: string) => void;
  clearErrorFeed: () => void;
}

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set) => ({
      currentResult: null,
      currentExplain: null,
      isAnalyzing: false,
      history: [],
      errorFeed: [],
      sidebarOpen: true,
      activeTab: "results",
      selectedTraceId: null,
      activeHeroState: "detect",

      setCurrentResult: (result) => set({ currentResult: result }),
      setCurrentExplain: (explain) => set({ currentExplain: explain }),
      setIsAnalyzing: (v) => set({ isAnalyzing: v }),

      addToHistory: (entry) =>
        set((state) => ({
          history: [entry, ...state.history].slice(0, 50),
        })),

      clearHistory: () => set({ history: [] }),
      setSidebarOpen: (open) => set({ sidebarOpen: open }),
      setActiveTab: (tab) => set({ activeTab: tab }),
      setSelectedTraceId: (id) => set({ selectedTraceId: id }),
      setActiveHeroState: (state) => set({ activeHeroState: state }),

      reset: () =>
        set({
          currentResult: null,
          currentExplain: null,
          isAnalyzing: false,
          activeTab: "results",
          selectedTraceId: null,
          activeHeroState: "detect",
        }),

      // Error feed — with duplicate protection
      addErrorEvent: (event) =>
        set((state) => {
          // Reject duplicate ID
          if (state.errorFeed.some((e) => e.id === event.id)) return {};
          // Reject duplicate trace_id (non-empty)
          if (
            event.trace_id &&
            state.errorFeed.some((e) => e.trace_id === event.trace_id)
          )
            return {};
          return { errorFeed: [event, ...state.errorFeed].slice(0, 100) };
        }),

      removeErrorEvent: (id) =>
        set((state) => ({
          errorFeed: state.errorFeed.filter((e) => e.id !== id),
        })),

      clearErrorFeed: () => set({ errorFeed: [] }),
    }),
    {
      name: "hallucisense-analysis",
      partialize: (state) => ({
        history: state.history,
        errorFeed: state.errorFeed,
      }),
    }
  )
);
