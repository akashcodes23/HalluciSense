import { api } from './api';

export const verificationService = {
  getReport: async (messageId: string) => {
    const response = await api.get(`/verification/${messageId}`);
    return response.data;
  },
  
  getSentenceDetail: async (messageId: string, sentenceIndex: number) => {
    const response = await api.get(`/verification/${messageId}/sentence/${sentenceIndex}`);
    return response.data;
  },
  
  verifyText: async (text: string) => {
    const response = await api.post('/verification/verify-text', { text });
    return response.data;
  },
};

// Mock data generator for Sprint 5 UI development (no live backend needed)
export const generateMockReport = (messageId: string) => ({
  id: 'report-' + messageId,
  message_id: messageId,
  overall_h_score: 0.42,
  overall_risk: 'NEEDS_VERIFICATION' as const,
  factual_error_score: 0.55,
  confidence_gap_score: 0.38,
  consistency_failure_score: 0.33,
  sentence_analyses: [
    {
      sentence_index: 0,
      sentence_text: 'The Earth orbits the Sun once every 365.25 days.',
      h_score: 0.05,
      risk_level: 'VERIFIED' as const,
      factual_error: 0.04,
      confidence_gap: 0.06,
      consistency_failure: 0.05,
      reasoning: 'Well-established astronomical fact corroborated by multiple sources.',
      evidence: [
        {
          claim: 'Earth orbit duration',
          snippet: 'Earth completes one orbit around the Sun every 365.25 days, which is known as a sidereal year.',
          source_name: 'Wikipedia: Earth',
          source_url: 'https://en.wikipedia.org/wiki/Earth',
          similarity_score: 0.97,
          is_supporting: true,
        }
      ]
    },
    {
      sentence_index: 1,
      sentence_text: 'The Moon was formed approximately 4.5 billion years ago from debris after a Mars-sized body collided with Earth.',
      h_score: 0.38,
      risk_level: 'NEEDS_VERIFICATION' as const,
      factual_error: 0.41,
      confidence_gap: 0.35,
      consistency_failure: 0.38,
      reasoning: 'The Giant Impact Hypothesis is widely accepted but the exact timing and parent body size are debated in literature.',
      evidence: [
        {
          claim: 'Moon formation giant impact',
          snippet: 'The giant-impact hypothesis proposes that the Moon formed from debris ejected when a Mars-sized protoplanet, called Theia, collided with the proto-Earth.',
          source_name: 'Wikipedia: Moon',
          source_url: 'https://en.wikipedia.org/wiki/Moon',
          similarity_score: 0.88,
          is_supporting: true,
        }
      ]
    },
    {
      sentence_index: 2,
      sentence_text: 'The first humans landed on Mars in 2024 during the Artemis IV mission.',
      h_score: 0.94,
      risk_level: 'LIKELY_HALLUCINATED' as const,
      factual_error: 0.97,
      confidence_gap: 0.89,
      consistency_failure: 0.96,
      reasoning: 'No human Mars landing has occurred. The Artemis program targets the Moon, not Mars.',
      evidence: [
        {
          claim: 'Human Mars landing 2024',
          snippet: 'As of 2024, no crewed missions to Mars have taken place. The Artemis program by NASA is focused on returning humans to the lunar surface.',
          source_name: 'Wikipedia: Artemis program',
          source_url: 'https://en.wikipedia.org/wiki/Artemis_program',
          similarity_score: 0.91,
          is_supporting: false,
        }
      ]
    }
  ]
});
