import { create } from 'zustand';

interface SentenceData {
  sentence_index: number;
  sentence_text: string;
  h_score: number;
  risk_level: 'VERIFIED' | 'NEEDS_VERIFICATION' | 'LIKELY_HALLUCINATED';
  factual_error: number;
  confidence_gap: number;
  consistency_failure: number;
  reasoning: string;
  evidence?: EvidenceItem[];
}

export interface EvidenceItem {
  claim: string;
  snippet: string;
  source_name: string;
  source_url: string;
  similarity_score: number;
  is_supporting: boolean;
}

export interface VerificationReport {
  id: string;
  message_id: string;
  overall_h_score: number;
  overall_risk: 'VERIFIED' | 'NEEDS_VERIFICATION' | 'LIKELY_HALLUCINATED';
  factual_error_score: number;
  confidence_gap_score: number;
  consistency_failure_score: number;
  sentence_analyses: SentenceData[];
}

interface UIState {
  isPanelOpen: boolean;
  panelWidth: number;
  activeMessageId: string | null;
  activeSentenceIndex: number | null;
  activeReport: VerificationReport | null;
  
  openPanel: (messageId: string, report: VerificationReport, sentenceIndex?: number) => void;
  closePanel: () => void;
  setActiveSentence: (index: number) => void;
  setPanelWidth: (width: number) => void;
}

export const useUIStore = create<UIState>((set) => ({
  isPanelOpen: false,
  panelWidth: 380,
  activeMessageId: null,
  activeSentenceIndex: null,
  activeReport: null,
  
  openPanel: (messageId, report, sentenceIndex = 0) => set({
    isPanelOpen: true,
    activeMessageId: messageId,
    activeReport: report,
    activeSentenceIndex: sentenceIndex,
  }),
  
  closePanel: () => set({
    isPanelOpen: false,
    activeSentenceIndex: null,
  }),
  
  setActiveSentence: (index) => set({ activeSentenceIndex: index }),
  
  setPanelWidth: (width) => set({ panelWidth: width }),
}));
