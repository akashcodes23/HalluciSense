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

interface AnalysisState {
  // Current analysis
  currentResult: AnalysisResponse | null;
  currentExplain: ExplainResponse | null;
  isAnalyzing: boolean;

  // History (persisted to localStorage)
  history: AnalysisHistoryEntry[];

  // UI state
  sidebarOpen: boolean;
  activeTab: "results" | "evidence" | "heatmap" | "trace";
  selectedTraceId: string | null;

  // Actions
  setCurrentResult: (result: AnalysisResponse | null) => void;
  setCurrentExplain: (explain: ExplainResponse | null) => void;
  setIsAnalyzing: (v: boolean) => void;
  addToHistory: (entry: AnalysisHistoryEntry) => void;
  clearHistory: () => void;
  setSidebarOpen: (open: boolean) => void;
  setActiveTab: (tab: AnalysisState["activeTab"]) => void;
  setSelectedTraceId: (id: string | null) => void;
  reset: () => void;
}

export const useAnalysisStore = create<AnalysisState>()(
  persist(
    (set) => ({
      currentResult: null,
      currentExplain: null,
      isAnalyzing: false,
      history: [],
      sidebarOpen: true,
      activeTab: "results",
      selectedTraceId: null,

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

      reset: () =>
        set({
          currentResult: null,
          currentExplain: null,
          isAnalyzing: false,
          activeTab: "results",
          selectedTraceId: null,
        }),
    }),
    {
      name: "hallucisense-analysis",
      partialize: (state) => ({ history: state.history }),
    }
  )
);
