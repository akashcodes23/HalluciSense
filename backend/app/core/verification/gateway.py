"""Phase 42 — Evidence Intelligence Gateway.

Routes claims to the appropriate verification modality:
- Symbolic Arithmetic Engine
- Unit Conversion Engine
- Temporal Date Engine
- Hybrid Textual Retriever + DeBERTa NLI Cross-Encoder
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.verification.claim_type_classifier import ClaimTypeClassifier
from app.core.verification.symbolic_verifier import evaluate_arithmetic_claim
from app.core.verification.unit_verifier import evaluate_unit_claim
from app.core.verification.temporal_verifier import evaluate_temporal_claim


class EvidenceIntelligenceGateway:
    """Master evidence intelligence dispatcher."""

    @staticmethod
    def verify_claim(claim_text: str) -> Dict[str, Any]:
        cls_info = ClaimTypeClassifier.classify(claim_text)
        claim_type = cls_info["claim_type"]
        
        # 1. Arithmetic
        if claim_type == "ARITHMETIC":
            sym_res = evaluate_arithmetic_claim(claim_text)
            if sym_res and sym_res.get("verified"):
                return {
                    "claim_text": claim_text,
                    "claim_type": claim_type,
                    "modality": "symbolic_arithmetic",
                    "status": "verified_symbolically",
                    "is_consistent": sym_res["is_consistent"],
                    "computed_value": sym_res.get("computed_value"),
                    "claimed_value": sym_res.get("claimed_value"),
                    "explanation": sym_res.get("explanation"),
                    "support_margin": 1.0 if sym_res["is_consistent"] else -1.0,
                    "entailment": 1.0 if sym_res["is_consistent"] else 0.0,
                    "contradiction": 0.0 if sym_res["is_consistent"] else 1.0,
                    "neutral": 0.0,
                }

        # 2. Unit Conversion
        if claim_type == "UNIT_CONVERSION":
            unit_res = evaluate_unit_claim(claim_text)
            if unit_res and unit_res.get("verified"):
                return {
                    "claim_text": claim_text,
                    "claim_type": claim_type,
                    "modality": "symbolic_unit",
                    "status": "verified_symbolically",
                    "is_consistent": unit_res["is_consistent"],
                    "expected_val": unit_res.get("expected_val"),
                    "claimed_val": unit_res.get("claimed_val"),
                    "explanation": unit_res.get("explanation"),
                    "support_margin": 1.0 if unit_res["is_consistent"] else -1.0,
                    "entailment": 1.0 if unit_res["is_consistent"] else 0.0,
                    "contradiction": 0.0 if unit_res["is_consistent"] else 1.0,
                    "neutral": 0.0,
                }

        # 3. Temporal Math
        if claim_type == "TEMPORAL_MATH":
            temp_res = evaluate_temporal_claim(claim_text)
            if temp_res and temp_res.get("verified"):
                return {
                    "claim_text": claim_text,
                    "claim_type": claim_type,
                    "modality": "symbolic_temporal",
                    "status": "verified_symbolically",
                    "is_consistent": temp_res["is_consistent"],
                    "expected_target": temp_res.get("expected_target"),
                    "explanation": temp_res.get("explanation"),
                    "support_margin": 1.0 if temp_res["is_consistent"] else -1.0,
                    "entailment": 1.0 if temp_res["is_consistent"] else 0.0,
                    "contradiction": 0.0 if temp_res["is_consistent"] else 1.0,
                    "neutral": 0.0,
                }

        # 4. Textual Fact (Forward to Retrieval + NLI)
        return {
            "claim_text": claim_text,
            "claim_type": "TEXTUAL_FACT",
            "modality": "retrieval_and_nli",
            "status": "requires_textual_grounding",
            "is_consistent": None,
            "explanation": "Evaluated via encyclopedic evidence retrieval and DeBERTa cross-encoder NLI.",
        }
