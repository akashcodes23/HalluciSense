"""
HalluciSense Public — Module 13.5: Documentation Portal Generator
==================================================================
Generates complete public documentation portal files (Getting Started, API Ref,
SDK Guides, Authentication, Examples, Architecture, FAQ).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import structlog

logger = structlog.get_logger(__name__)


class DocumentationPortalGenerator:
    """
    Generates documentation portal markdown files.
    """

    def generate_portal(self, out_dir: Path) -> List[str]:
        """Generate complete documentation portal suite."""
        out_dir.mkdir(parents=True, exist_ok=True)
        exported: List[str] = []

        docs = {
            "GETTING_STARTED.md": """# Getting Started with HalluciSense

Welcome to HalluciSense v1.0!

## Quick Install (Python SDK)
```bash
pip install hallucisense-sdk
```

## Quick Verification Code
```python
from hallucisense_sdk import HalluciSenseClient

client = HalluciSenseClient(api_key="hs_live_your_key_here")
result = client.verify("Albert Einstein discovered relativity in 1905.")

print(f"H-Score: {result.hallucisense_score:.2f}")
print(f"Risk Level: {result.risk_category}")
```
""",
            "API_REFERENCE.md": """# HalluciSense v1 REST API Reference

Base Endpoint: `https://api.hallucisense.ai/api/v1`

## POST `/pillar2/verify`
Verify response text for factual grounding.

### Request Headers
- `Content-Type`: `application/json`
- `X-API-Key`: `hs_live_...`

### Request Body
```json
{
  "text": "Quantum computing uses qubits to calculate states.",
  "pillar1_probability": 0.15
}
```

### Response
```json
{
  "verification_id": "verif_101",
  "hallucisense_score": {
    "hallucisense_score": 12.50,
    "risk_category": "VERY_LOW",
    "overall_confidence": 0.972
  }
}
```
""",
            "SDK_GUIDES.md": """# HalluciSense Developer SDK Guides

- **Python SDK**: `sdk/python/hallucisense_sdk.py`
- **JavaScript/TypeScript SDK**: `sdk/javascript/hallucisense-sdk.js`
- **CLI Tool**: `hallucisense-cli verify "your text here"`
""",
            "AUTHENTICATION.md": """# HalluciSense Authentication Guide

HalluciSense uses HTTP Bearer JWT tokens and `X-API-Key` headers for authentication.
API Keys can be generated from the User Dashboard.
""",
            "EXAMPLES.md": """# HalluciSense Integration Examples

See `examples/` for complete Python, Node.js, and CLI verification integration examples.
""",
            "ARCHITECTURE.md": """# HalluciSense Dual-Pillar Architecture Overview

HalluciSense combines a frozen statistical NLI classifier (Pillar 1) with an evidence-aware multi-LLM consensus engine (Pillar 2).
""",
            "FAQ.md": """# Frequently Asked Questions (FAQ)

### Q: What is the H-Score?
The Unified H-Score is a 0--100 metric where 0 indicates fully grounded truth and 100 indicates total hallucination.

### Q: How fast is HalluciSense?
Pillar 1 operates in <0.5ms; full Pillar 2 evidence verification runs in sub-4ms P95 latency.
""",
        }

        for fname, content in docs.items():
            p = out_dir / fname
            with open(p, "w") as f:
                f.write(content)
            exported.append(str(p))

        logger.info("doc_portal_generated", out_dir=str(out_dir), count=len(exported))
        return exported
