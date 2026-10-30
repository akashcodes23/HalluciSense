import assert from "node:assert";
import {
  normalizeExplainResponse,
  normalizePredictResponse,
  adaptCanonicalAnalysisResponse,
} from "../hallucisense-api.ts";
import type { HalluciSenseExplainResponse } from "../../types/verification-types.ts";

console.log("Running Phase 16A Frontend Contract Normalization Tests...");

// 1. Legacy Explain Response Normalization
const legacyExplainMock: HalluciSenseExplainResponse = {
  source: "test",
  response_text: "Water is H2O",
  latency_ms: 120,
  prediction: {
    is_hallucinated: false,
    hallucination_probability: 0.15,
    operating_threshold: 0.54,
    claim_count: 1,
    claims: ["Water is H2O"],
    confidence_score: 0.85,
  },
  explanation: {
    summary: "Legacy analysis test summary",
    primary_driver: "Pillar 1 Retrieval",
    verdict: "VERIFIED",
    risk_severity: "LOW",
    claim_analysis: [],
    structural_analysis: {},
    recommendation: "Verified by external reference",
    pillar_contributions: {
      pillar_1_probability: 0.85,
      pillar_2_probability: 0.15,
      probability_disagreement: 0.70,
    },
  },
  semantic_grounding: {
    status: "grounded",
    claims: [
      {
        claim_id: 1,
        claim_text: "Water is H2O",
        evidence_count: 1,
        primary_status: "entailment",
        max_entailment: 0.95,
        mean_contradiction: 0.01,
        support_margin: 0.94,
        evidence_details: [
          {
            title: "Water",
            url: "https://en.wikipedia.org/wiki/Water",
            snippet: "Water has formula H2O",
            entailment: 0.95,
            neutral: 0.04,
            contradiction: 0.01,
            label: "entailment",
            confidence: 0.95,
          },
        ],
      },
    ],
  },
};

const normLegacy = normalizeExplainResponse(legacyExplainMock, 120);

// Invariant 1: Legacy P2 only populates P2
assert.strictEqual(
  normLegacy.fusion_contribution?.pillar2_confidence_contribution,
  0.15,
  "Legacy P2 must populate pillar2_confidence_contribution"
);

// Invariant 2: Legacy responses MUST NEVER fabricate P3 (must be null)
assert.strictEqual(
  normLegacy.fusion_contribution?.pillar3_consistency_contribution,
  null,
  "Legacy response must never fabricate P3; must remain null"
);

// Invariant 3: P1 populated correctly
assert.strictEqual(
  normLegacy.fusion_contribution?.pillar1_retrieval_contribution,
  0.85,
  "P1 contribution should match legacy P1"
);

// 2. Legacy Explain Response with Missing P2
const legacyExplainMissingP2 = {
  explanation: {
    summary: "Missing P2 summary",
    pillar_contributions: {
      pillar_1_probability: 0.72,
    },
  },
} as unknown as HalluciSenseExplainResponse;

const normMissing = normalizeExplainResponse(legacyExplainMissingP2, 80);
assert.strictEqual(
  normMissing.fusion_contribution?.pillar2_confidence_contribution,
  null,
  "Missing P2 must remain null, never fabricated as 0"
);
assert.strictEqual(
  normMissing.fusion_contribution?.pillar3_consistency_contribution,
  null,
  "P3 must remain null when missing"
);

// 3. Canonical Adapter Test
const canonicalRaw = {
  trace_id: "TRACE_1234",
  overall_h_score: 0.12,
  risk_level: "VERIFIED",
  confidence: 0.88,
  pillar_scores: {
    retrieval: 0.12,
    confidence: null,
    consistency: null,
  },
  processing_time_ms: 95,
  sentence_scores: [
    {
      sentence_index: 0,
      text: "The Earth orbits the Sun.",
      score: 0.12,
      risk_level: "VERIFIED",
    },
  ],
  evidence: [
    {
      id: "ev_0",
      title: "Earth",
      snippet: "Earth orbits the Sun once per year.",
      score: 0.98,
      source: "Wikipedia: Earth",
      url: "https://en.wikipedia.org/wiki/Earth",
    },
  ],
};

const canonicalAdapted = adaptCanonicalAnalysisResponse(canonicalRaw as any, 95);
assert.strictEqual(canonicalAdapted.trace_id, "TRACE_1234");
assert.strictEqual(canonicalAdapted.overall_h_score, 0.12);
assert.strictEqual(canonicalAdapted.pillar_scores.retrieval, 0.12);
assert.strictEqual(canonicalAdapted.pillar_scores.confidence, null, "Canonical missing P2 remains null");
assert.strictEqual(canonicalAdapted.pillar_scores.consistency, null, "Canonical missing P3 remains null");
assert.strictEqual(canonicalAdapted.pillar_status?.p1_status, "EXECUTED");
assert.strictEqual(canonicalAdapted.pillar_status?.p2_status, "UNAVAILABLE");
assert.strictEqual(canonicalAdapted.pillar_status?.p3_status, "UNAVAILABLE");

console.log("Phase 16A Frontend Contract Normalization Tests: ALL PASSED!");
