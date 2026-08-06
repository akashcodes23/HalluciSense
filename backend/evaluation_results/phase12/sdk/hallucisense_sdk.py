"""
HalluciSense Python SDK Client
==============================
Official Python client library for HalluciSense Hallucination Verification API.
Usage:
    from hallucisense_sdk import HalluciSenseClient

    client = HalluciSenseClient(api_key="hs_live_...")
    result = client.verify("Albert Einstein published relativity papers in 1905.")
    print(result.hallucisense_score, result.risk_category)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import urllib.request
import json


class VerificationResult:
    def __init__(self, data: Dict[str, Any]):
        self.raw = data
        self.verification_id = data.get("verification_id", "")
        self.text = data.get("text", "")
        self.execution_time_ms = data.get("execution_time_ms", 0.0)

        hs = data.get("hallucisense_score", {})
        self.hallucisense_score = hs.get("hallucisense_score", 0.0)
        self.risk_category = hs.get("risk_category", "UNKNOWN")
        self.confidence = hs.get("overall_confidence", 0.0)
        self.pillar1_probability = hs.get("pillar1_probability", 0.0)


class HalluciSenseClient:
    """Official HalluciSense Python SDK Client."""

    def __init__(self, api_key: str, base_url: str = "http://localhost:8000/api/v1/pillar2"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def verify(self, text: str, pillar1_prob: float = 0.50) -> VerificationResult:
        """
        Verify input text for hallucinations using Pillar 2 engine.

        Parameters
        ----------
        text : str
        pillar1_prob : float

        Returns
        -------
        VerificationResult
        """
        url = f"{self.base_url}/verify"
        payload = json.dumps({"text": text, "pillar1_probability": pillar1_prob}).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "User-Agent": "HalluciSense-Python-SDK/10.0.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return VerificationResult(data)
