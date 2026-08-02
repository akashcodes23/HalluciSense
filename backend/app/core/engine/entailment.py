"""
Natural Language Inference engine for HalluciSense Pillar 1.

Determines whether retrieved evidence:
- entails a generated claim,
- contradicts it, or
- is neutral / insufficient.

Evidence is the NLI premise.
Generated claim is the NLI hypothesis.
"""

from typing import Dict, List
import torch
import structlog
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = structlog.get_logger(__name__)


class EvidenceEntailmentEngine:
    """
    NLI-based factual verification engine.

    premise    = retrieved evidence
    hypothesis = generated factual claim
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small"
    ):
        logger.info(
            "initializing_nli_model",
            model_name=model_name
        )

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.eval()

        # Apple Silicon acceleration when available.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # Cache id2label mapping during initialization
        self.label_map: Dict[str, int] = {}
        id2label = getattr(self.model.config, "id2label", {})
        for idx, label in id2label.items():
            label_str = str(label).lower()
            if "entail" in label_str:
                self.label_map["entailment"] = int(idx)
            elif "neutral" in label_str:
                self.label_map["neutral"] = int(idx)
            elif "contrad" in label_str:
                self.label_map["contradiction"] = int(idx)

        logger.info(
            "nli_model_initialized",
            model_name=model_name,
            device=str(self.device)
        )

    def classify(
        self,
        claim: str,
        evidence: str
    ) -> Dict[str, float]:
        """
        Classify the relationship between evidence and claim.

        Returns:
        {
            "entailment": float,
            "neutral": float,
            "contradiction": float
        }
        """
        return self.classify_batch([claim], [evidence])[0]

    def classify_batch(
        self,
        claims: List[str],
        evidences: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """
        Classify a batch of relationship pairs (claims and evidences).

        Args:
            claims: List of generated claim strings (hypotheses).
            evidences: List of retrieved evidence strings (premises).
            batch_size: Number of pairs per forward pass.

        Returns:
            List of dicts containing entailment, neutral, and contradiction scores.
        """
        if len(claims) != len(evidences):
            raise ValueError(
                f"Claims and evidences length mismatch: {len(claims)} vs {len(evidences)}"
            )

        if not claims:
            return []

        results: List[Dict[str, float]] = [
            {
                "entailment": 0.0,
                "neutral": 1.0,
                "contradiction": 0.0
            }
            for _ in range(len(claims))
        ]

        valid_indices: List[int] = []
        valid_evidences: List[str] = []
        valid_claims: List[str] = []

        for idx, (c, e) in enumerate(zip(claims, evidences)):
            if c and e and c.strip() and e.strip():
                valid_indices.append(idx)
                valid_evidences.append(e)
                valid_claims.append(c)

        if not valid_indices:
            return results

        ent_idx = self.label_map.get("entailment", 0)
        neu_idx = self.label_map.get("neutral", 1)
        con_idx = self.label_map.get("contradiction", 2)

        num_valid = len(valid_indices)

        for b_start in range(0, num_valid, batch_size):
            b_end = min(b_start + batch_size, num_valid)
            b_evidences = valid_evidences[b_start:b_end]
            b_claims = valid_claims[b_start:b_end]

            inputs = self.tokenizer(
                b_evidences,
                b_claims,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            probs_cpu = probs.cpu().numpy()

            for offset, orig_idx in enumerate(valid_indices[b_start:b_end]):
                row = probs_cpu[offset]
                results[orig_idx] = {
                    "entailment": float(row[ent_idx]),
                    "neutral": float(row[neu_idx]),
                    "contradiction": float(row[con_idx])
                }

        return results