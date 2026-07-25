export const MODELS = [
  { id: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro', provider: 'google' },
  { id: 'gemini-1.5-flash', label: 'Gemini 1.5 Flash', provider: 'google' },
  { id: 'gpt-4o', label: 'GPT-4o', provider: 'openai' },
  { id: 'gpt-4o-mini', label: 'GPT-4o Mini', provider: 'openai' },
] as const;

export const DEFAULT_MODEL = 'gemini-1.5-flash';

export const APP_NAME = 'HalluciSense';
export const APP_TAGLINE = 'Confidence-Aware AI — Know When to Trust';

export const RISK_LABELS = {
  VERIFIED: 'Verified',
  NEEDS_VERIFICATION: 'Uncertain',
  LIKELY_HALLUCINATED: 'Hallucinated',
} as const;
