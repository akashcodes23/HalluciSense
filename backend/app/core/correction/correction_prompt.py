"""Correction Prompts for HalluciSense Phase 11.

Enforces strict evidence grounding, prevents fabricated citations, preserves supported claims,
and repairs contradicted or unsupported claims using only verified evidence passages.
"""

from __future__ import annotations
from typing import List, Dict, Any


SYSTEM_CORRECTION_PROMPT = """You are the HalluciSense Closed-Loop Correction Engine.

Your sole purpose is to repair factual errors, numerical conflicts, unit/scale mismatches,
negation inversions, and unsupported claims in an AI-generated response.

STRICT OPERATIONAL RULES:
1. You MUST preserve all claims that HalluciSense marked as SUPPORTED.
2. You MUST remove, replace, or qualify claims marked as CONTRADICTED or UNSUPPORTED.
3. You MUST use the supplied evidence as your sole factual authority.
4. Do NOT introduce facts that are absent from the supplied evidence unless they are necessary grammatical glue.
5. Do NOT invent citations, URLs, author names, or paper titles.
6. Do NOT fabricate numerical values, measurements, dates, or mechanisms.
7. If evidence is insufficient to establish a claim, explicitly qualify that the claim could not be independently verified.
8. Maintain a concise, professional, and authoritative scientific tone.
"""


def build_claim_correction_prompt(
    user_query: str,
    original_text: str,
    claims_verification: List[Dict[str, Any]],
    retrieved_evidence: List[Dict[str, Any]],
) -> str:
    """Builds a structured prompt containing atomic claims and verified evidence."""
    evidence_lines = []
    for i, ev in enumerate(retrieved_evidence, 1):
        title = ev.get("source_name") or ev.get("source_title") or "Authoritative Scientific Reference"
        snip = ev.get("snippet") or ev.get("excerpt") or ev.get("claim") or ""
        evidence_lines.append(f"[{i}] ({title}): {snip}")
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "No external evidence available."

    claims_lines = []
    for c in claims_verification:
        cid = c.get("claim_id", "claim")
        text = c.get("claim_text", "")
        status = c.get("status", "UNCERTAIN")
        err = c.get("error_type", "NONE")
        claims_lines.append(f"- [{cid}] ({status} | Error: {err}): \"{text}\"")
    claims_block = "\n".join(claims_lines)

    return f"""USER QUESTION:
{user_query}

ORIGINAL GENERATED RESPONSE:
{original_text}

VERIFIED ATOMIC CLAIMS:
{claims_block}

RETRIEVED AUTHORITATIVE EVIDENCE:
{evidence_block}

TASK:
Produce a corrected response that fixes all CONTRADICTED or UNSUPPORTED claims using the retrieved evidence, while preserving all SUPPORTED claims exactly.
"""
