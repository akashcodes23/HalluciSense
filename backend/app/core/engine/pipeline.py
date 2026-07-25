import re
from typing import List, Optional
from .types import (
    HallucinationReport,
    SentenceAnalysis,
    TokenAnalysis,
    EvidenceItem,
    RiskLevel
)
from .pillar1_retrieval import Pillar1RetrievalEngine
from .pillar2_confidence import Pillar2ConfidenceEngine
from .pillar3_consistency import Pillar3ConsistencyEngine
from .fusion import FusionEngine

class HallucinationDetectionPipeline:
    """
    Master Hybrid Hallucination Detection Pipeline Orchestrator.
    Executes Pillar 1, Pillar 2, Pillar 3, and performs Fusion at sentence & document levels.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ):
        self.p1_engine = Pillar1RetrievalEngine()
        self.p2_engine = Pillar2ConfidenceEngine()
        self.p3_engine = Pillar3ConsistencyEngine()
        self.fusion_engine = FusionEngine(alpha=alpha, beta=beta, gamma=gamma)

    def _split_sentences(self, text: str) -> List[tuple[str, int, int]]:
        """
        Split text into sentence strings along with character start and end offsets.
        """
        sentence_spans = []
        # Regex for terminal punctuation followed by space or end-of-string
        pattern = re.compile(r'[^.!?]+[.!?]+|\s*[^.!?]+$')
        
        start = 0
        for match in pattern.finditer(text):
            sentence_text = match.group(0).strip()
            if sentence_text:
                span_start = text.find(sentence_text, start)
                span_end = span_start + len(sentence_text)
                sentence_spans.append((sentence_text, span_start, span_end))
                start = span_end

        if not sentence_spans and text.strip():
            sentence_spans.append((text.strip(), 0, len(text.strip())))

        return sentence_spans

    def analyze_response(
        self,
        full_text: str,
        token_probabilities: Optional[List[float]] = None,
        evidence_items: Optional[List[EvidenceItem]] = None,
        sample_responses: Optional[List[str]] = None
    ) -> HallucinationReport:
        """
        Run complete hybrid hallucination detection pipeline on an LLM response.
        """
        clean_text = full_text.strip()
        if not clean_text:
            p1_res = self.p1_engine.analyze("", [])
            p2_res = self.p2_engine.analyze([], [])
            p3_res = self.p3_engine.analyze("", [])
            h_score, risk, _, weights = self.fusion_engine.fuse(p1_res, p2_res, p3_res)
            return HallucinationReport(
                full_text="",
                overall_h_score=0.0,
                overall_risk_level=RiskLevel.VERIFIED,
                sentence_analyses=[],
                token_analyses=[],
                pillar1_summary=p1_res,
                pillar2_summary=p2_res,
                pillar3_summary=p3_res,
                weights_used=weights
            )

        if evidence_items is None:
            evidence_items = []
        if sample_responses is None:
            sample_responses = []

        # 1. Document-wide Pillar Analyses
        p1_global = self.p1_engine.analyze(clean_text, evidence_items)
        
        # Tokenize response simple whitespace/word-boundary split
        raw_tokens = re.findall(r'\S+', clean_text)
        token_analyses, _, _, _ = self.p2_engine.evaluate_tokens(raw_tokens, token_probabilities)
        p2_global = self.p2_engine.analyze(raw_tokens, token_probabilities)
        
        p3_global = self.p3_engine.analyze(clean_text, sample_responses)

        # 2. Sentence-level granular analysis
        sentence_spans = self._split_sentences(clean_text)
        sentence_analyses: List[SentenceAnalysis] = []

        for idx, (sent_text, s_start, s_end) in enumerate(sentence_spans):
            # Extract sentence specific evidence
            sent_p1 = self.p1_engine.analyze(sent_text, evidence_items)
            
            # Extract tokens belonging to this sentence span
            sent_tokens = re.findall(r'\S+', sent_text)
            sent_probs = token_probabilities[:len(sent_tokens)] if token_probabilities else None
            sent_p2 = self.p2_engine.analyze(sent_tokens, sent_probs)

            # Sentence level consistency
            sent_p3 = self.p3_engine.analyze(sent_text, sample_responses)

            # Fuse for sentence H-Score
            s_h_score, s_risk, s_color, _ = self.fusion_engine.fuse(sent_p1, sent_p2, sent_p3)

            sentence_analyses.append(
                SentenceAnalysis(
                    sentence_id=idx,
                    text=sent_text,
                    start_char=s_start,
                    end_char=s_end,
                    factual_error=sent_p1.factual_error_score,
                    confidence_gap=sent_p2.confidence_gap_score,
                    consistency_failure=sent_p3.consistency_failure_score,
                    hallucination_score=s_h_score,
                    risk_level=s_risk,
                    color_code=s_color,
                    evidence=sent_p1.evidence,
                    reasoning=f"H-Score: {s_h_score:.2f}. {sent_p1.reasoning} {sent_p2.reasoning}"
                )
            )

        # 3. Overall document level H-Score fusion
        overall_h_score, overall_risk, _, weights = self.fusion_engine.fuse(
            p1_global, p2_global, p3_global
        )

        return HallucinationReport(
            full_text=clean_text,
            overall_h_score=overall_h_score,
            overall_risk_level=overall_risk,
            sentence_analyses=sentence_analyses,
            token_analyses=token_analyses,
            pillar1_summary=p1_global,
            pillar2_summary=p2_global,
            pillar3_summary=p3_global,
            weights_used=weights
        )
