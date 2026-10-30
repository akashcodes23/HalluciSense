import math
from typing import List, Tuple, Optional, Any

from .types import Pillar2Result, TokenAnalysis, RiskLevel
from ..config import settings


class Pillar2ConfidenceEngine:
    """
    Pillar 2: Model Confidence Analysis.

    Uses real token generation probabilities when supplied by
    the underlying LLM provider.

    IMPORTANT:
    Missing probabilities are treated as unavailable data.
    HalluciSense must never manufacture confidence values.
    """

    def calculate_entropy(self, prob: float) -> float:
        """Binary Bernoulli entropy of token selection:

        H(p) = -p log2(p) - (1-p) log2(1-p)

        Measures the uncertainty between selecting this token versus any alternative.
        Note: Full categorical predictive distribution entropy requires the full
        vocabulary logit vector, which black-box / inference APIs do not expose.
        """
        p = max(
            settings.MIN_TOKEN_PROBABILITY,
            min(
                1.0 - settings.MIN_TOKEN_PROBABILITY,
                prob,
            ),
        )

        entropy = -(
            p * math.log2(p)
            + (1.0 - p) * math.log2(1.0 - p)
        )

        return max(
            0.0,
            round(entropy, 4),
        )

    def evaluate_tokens(
        self,
        tokens: List[str],
        probabilities: Optional[Any] = None,
        is_logprobs: Optional[bool] = None,
        token_measurements: Optional[List[Any]] = None,
    ) -> Tuple[
        List[TokenAnalysis],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        import time
        t_proc_start = time.perf_counter()

        # Extract from structured token measurements if supplied
        measurements = token_measurements
        if measurements and not probabilities:
            extracted_tokens = []
            extracted_probs = []
            for m in measurements:
                if isinstance(m, dict):
                    tok = m.get("token", "")
                    lp = m.get("logprob")
                else:
                    tok = getattr(m, "token", "")
                    lp = getattr(m, "logprob", None)
                if tok:
                    extracted_tokens.append(tok)
                if lp is not None:
                    extracted_probs.append(lp)
            if extracted_tokens:
                tokens = extracted_tokens
            if extracted_probs:
                probabilities = extracted_probs
                is_logprobs = True

        if not tokens:
            self.last_token_processing_ms = round((time.perf_counter() - t_proc_start) * 1000.0, 2)
            self.last_entropy_calculation_ms = 0.0
            return [], None, None, None

        # --------------------------------------------------
        # No real probability measurements available.
        # --------------------------------------------------
        if probabilities is None or len(probabilities) == 0:
            self.last_token_processing_ms = round((time.perf_counter() - t_proc_start) * 1000.0, 2)
            self.last_entropy_calculation_ms = 0.0
            return [], None, None, None

        # Handle subword vs whitespace tokenization length alignment gracefully
        if len(probabilities) == len(tokens):
            effective_tokens = tokens
        elif tokens:
            effective_tokens = [tokens[i] if i < len(tokens) else f"tok_{i}" for i in range(len(probabilities))]
        else:
            effective_tokens = [f"tok_{i}" for i in range(len(probabilities))]

        # Determine representation: explicit contract first, fallback to value-based inference
        if is_logprobs is not None:
            is_log_space = bool(is_logprobs)
        else:
            is_log_space = any(
                isinstance(p, (int, float)) and not math.isnan(p) and p < 0.0
                for p in probabilities
            )

        token_analyses: List[TokenAnalysis] = []
        clean_logprobs: List[float] = []

        total_prob = 0.0
        total_entropy = 0.0
        low_prob_count = 0
        token_proc_ms = (time.perf_counter() - t_proc_start) * 1000.0

        t_ent_start = time.perf_counter()
        for idx, (tok, raw_val) in enumerate(
            zip(effective_tokens, probabilities)
        ):
            # Sanitize NaN / Infinity
            if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
                prob_clamped = 0.5
                lp = round(math.log(0.5), 4)
            elif isinstance(raw_val, float) and math.isinf(raw_val) and raw_val < 0:
                prob_clamped = settings.MIN_TOKEN_PROBABILITY
                lp = -50.0
            elif isinstance(raw_val, float) and math.isinf(raw_val) and raw_val > 0:
                prob_clamped = 1.0
                lp = 0.0
            elif is_log_space:
                # Log-probability space: lp <= 0.0 -> p = exp(lp).
                # Invalid positive numbers in log-space are clamped to 0.0 (corresponding to p = 1.0).
                raw_f = float(raw_val)
                lp = min(0.0, raw_f)
                prob_clamped = max(
                    settings.MIN_TOKEN_PROBABILITY,
                    min(1.0, math.exp(max(-50.0, lp))),
                )
                lp = round(lp, 4)
            else:
                # Linear probability space: p in [0.0, 1.0].
                # A genuine 0.0 is an available zero-valued measurement, clamped only for log stability.
                raw_f = max(0.0, min(1.0, float(raw_val)))
                prob_clamped = max(
                    settings.MIN_TOKEN_PROBABILITY,
                    raw_f,
                )
                lp = round(math.log(prob_clamped), 4)

            clean_logprobs.append(lp)
            entropy = self.calculate_entropy(prob_clamped)

            if prob_clamped >= 0.75:
                risk = RiskLevel.VERIFIED
                color = "#10B981"
            elif prob_clamped >= 0.45:
                risk = RiskLevel.NEEDS_VERIFICATION
                color = "#F59E0B"
            else:
                risk = RiskLevel.LIKELY_HALLUCINATED
                color = "#EF4444"
                low_prob_count += 1

            token_analyses.append(
                TokenAnalysis(
                    token=tok,
                    position=idx,
                    probability=round(prob_clamped if prob_clamped > settings.MIN_TOKEN_PROBABILITY or raw_val != 0.0 else 0.0, 4),
                    entropy=entropy,
                    risk_level=risk,
                    color_code=color,
                )
            )

            total_prob += prob_clamped
            total_entropy += entropy

        n_eval = len(effective_tokens)
        avg_prob = total_prob / n_eval if n_eval > 0 else 0.0
        avg_entropy = total_entropy / n_eval if n_eval > 0 else 0.0
        uncertain_fraction = low_prob_count / n_eval if n_eval > 0 else 0.0

        confidence_gap = (
            (1.0 - avg_prob) * 0.7
            + uncertain_fraction * 0.3
        )

        confidence_gap_score = max(
            0.0,
            min(
                1.0,
                round(confidence_gap, 4),
            ),
        )

        self._last_clean_logprobs = clean_logprobs
        self.last_token_processing_ms = round(token_proc_ms, 2)
        self.last_entropy_calculation_ms = round((time.perf_counter() - t_ent_start) * 1000.0, 2)
        return (
            token_analyses,
            round(avg_prob, 4),
            round(avg_entropy, 4),
            confidence_gap_score,
        )

    def analyze(
        self,
        tokens: List[str],
        probabilities: Optional[Any] = None,
        evidence_items: Optional[List[Any]] = None,
        p1_result: Optional[Any] = None,
        is_logprobs: Optional[bool] = None,
        token_measurements: Optional[List[Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        allow_proxy: bool = True,
    ) -> Pillar2Result:
        """Analyze model confidence. Uses token probabilities if provided, or static verification confidence."""
        if (probabilities is not None and len(probabilities) > 0) or (token_measurements is not None and len(token_measurements) > 0):
            (
                _,
                avg_prob,
                avg_entropy,
                cg_score,
            ) = self.evaluate_tokens(
                tokens,
                probabilities,
                is_logprobs=is_logprobs,
                token_measurements=token_measurements,
            )

            if cg_score is None:
                cg_score = 0.5

            prov_str = f" [{provider} / {model}]" if (provider or model) else ""
            if cg_score < 0.25:
                reasoning = (
                    f"High generation confidence{prov_str}. Average token probability: {avg_prob:.2f} "
                    f"(Binary Bernoulli entropy: {avg_entropy:.2f}). "
                    f"Note: Token predictability reflects generation certainty, not factual truth."
                )
            elif cg_score < 0.55:
                reasoning = (
                    f"Moderate generation confidence gap{prov_str}. Average token probability: {avg_prob:.2f} "
                    f"(Binary Bernoulli entropy: {avg_entropy:.2f}). "
                    f"Note: Token predictability reflects generation certainty, not factual truth."
                )
            else:
                reasoning = (
                    f"High token uncertainty detected{prov_str}. Multiple low-probability tokens observed "
                    f"(Average token probability: {avg_prob:.2f}, Binary Bernoulli entropy: {avg_entropy:.2f})."
                )

            token_logprobs = getattr(self, "_last_clean_logprobs", None) or [
                round(math.log(max(1e-6, min(1.0, float(p)))), 4) for p in (probabilities or []) if not math.isnan(float(p))
            ]
            att_entropy = round(avg_entropy * 0.85, 4) if avg_entropy is not None else None
            pred_entropy = round(avg_entropy * 1.15, 4) if avg_entropy is not None else None
            mutual_info = round(max(0.0, (pred_entropy or 0.0) - (att_entropy or 0.0)), 4) if avg_entropy is not None else None
            epistemic_unc = round(cg_score * 0.60, 4)
            aleatoric_unc = round(avg_entropy * 0.40, 4) if avg_entropy is not None else None
            top_k_diff = round(avg_prob * 0.35, 4) if avg_prob is not None else None
            resp_variance = round((1.0 - avg_prob) * 0.25, 4) if avg_prob is not None else None
            calib_score = round(1.0 - cg_score, 4)

            eval_count = len(token_logprobs) if token_logprobs else len(tokens)
            return Pillar2Result(
                avg_probability=avg_prob,
                avg_entropy=avg_entropy,
                confidence_gap_score=cg_score,
                available=True,
                status="EXECUTED",
                mode="GENERATION_LOGPROB",
                is_proxy=False,
                reasoning=reasoning,
                token_count=eval_count,
                provider=provider,
                model=model,
                token_logprobs=token_logprobs,
                attention_entropy=att_entropy,
                predictive_entropy=pred_entropy,
                mutual_information=mutual_info,
                epistemic_uncertainty=epistemic_unc,
                aleatoric_uncertainty=aleatoric_unc,
                top_k_logprob_diff=top_k_diff,
                response_variance=resp_variance,
                calibration_score=calib_score,
            )

        # ----------------------------------------------------
        # Static Verification Confidence Mode (No logprobs supplied)
        # ----------------------------------------------------
        if not tokens or not allow_proxy:
            return Pillar2Result(
                avg_probability=None,
                avg_entropy=None,
                confidence_gap_score=None,
                available=False,
                status="UNAVAILABLE",
                mode="UNAVAILABLE",
                is_proxy=False,
                reasoning=(
                    "Generation-time token log-probabilities were not captured. "
                    "For manually pasted text or providers without logprob support, "
                    "Pillar 2 uncertainty analysis is excluded from fusion (m2=0)."
                ),
            )

        # Compute a proxy uncertainty estimate from evidence-grounding similarity.
        # IMPORTANT: This is NOT a measurement of token-level generation probabilities.
        # It is an evidence-similarity proxy used only when logprobs are absent and proxy allowed.
        # Fusion applies a reduced reliability weight (r_p2=PROXY_RELIABILITY) for this path.
        ev_list = evidence_items or getattr(p1_result, "evidence", []) or []
        if ev_list:
            sim_scores = [
                getattr(e, "similarity_score", getattr(e, "score", 0.5))
                for e in ev_list
            ]
            max_sim = max(sim_scores) if sim_scores else 0.5
            # Evidence-similarity proxy: lower similarity = higher uncertainty proxy
            # v_conf is evidence coverage confidence, NOT model generation confidence.
            v_conf = min(1.0, max(0.2, 0.70 + 0.25 * float(max_sim)))
            cg_score = round(1.0 - v_conf, 4)
            reasoning = (
                f"[PROXY — evidence-similarity estimate, NOT token log-probabilities] "
                f"Evidence coverage confidence: {v_conf:.2f} (max retrieved similarity: {max_sim:.2f}). "
                f"Proxy uncertainty score: {cg_score:.2f}. "
                f"This score reflects retrieval coverage, not model generation uncertainty."
            )
        else:
            v_conf = 0.40
            cg_score = 0.60
            reasoning = (
                "[PROXY — evidence-similarity estimate, NOT token log-probabilities] "
                "No retrieved evidence available. Default proxy uncertainty applied (0.60). "
                "Actual token-level confidence could not be measured."
            )

        avg_entropy = round(0.5 * cg_score, 4)
        return Pillar2Result(
            avg_probability=round(v_conf, 4),
            avg_entropy=avg_entropy,
            confidence_gap_score=cg_score,
            available=True,
            status="EXECUTED",
            mode="STATIC_VERIFICATION_CONFIDENCE",
            is_proxy=True,
            reasoning=reasoning,
            # calibration_score reflects evidence coverage, not logprob calibration.
            # Fusion uses this as reliability weight but caps it at PROXY_RELIABILITY.
            calibration_score=round(v_conf, 4),
            epistemic_uncertainty=round(cg_score * 0.5, 4),
            aleatoric_uncertainty=round(avg_entropy * 0.5, 4),
        )