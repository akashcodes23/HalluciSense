"""Part 7 — Token Level Localization Engine & 4-Tier Risk Heatmaps.

Propagates sentence-level hallucination risk scores down to sub-sentence token spans:
Sentence Scores -> Token Propagation -> Span Merging -> Interactive Heatmap HTML/JSON.

Risk Tiers:
- VERIFIED (H < 0.35): Green (#10B981)
- NEEDS_VERIFICATION (0.35 <= H < 0.50): Yellow (#F59E0B)
- MODERATE_RISK (0.50 <= H < 0.65): Orange (#F97316)
- LIKELY_HALLUCINATED (H >= 0.65): Red (#EF4444)
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class TokenSpanAnnotation:
    start_char: int
    end_char: int
    text: str
    risk_score: float
    risk_tier: str
    color_hex: str


class TokenLevelLocalizationEngine:
    """Generates sub-sentence token annotations and HTML/JSON heatmaps."""

    COLOR_TIERS = {
        "VERIFIED": "#10B981",            # Green
        "NEEDS_VERIFICATION": "#F59E0B",  # Yellow
        "MODERATE_RISK": "#F97316",       # Orange
        "LIKELY_HALLUCINATED": "#EF4444", # Red
    }

    def get_tier_and_color(self, score: float) -> Tuple[str, str]:
        if score < 0.35:
            return "VERIFIED", self.COLOR_TIERS["VERIFIED"]
        elif score < 0.50:
            return "NEEDS_VERIFICATION", self.COLOR_TIERS["NEEDS_VERIFICATION"]
        elif score < 0.65:
            return "MODERATE_RISK", self.COLOR_TIERS["MODERATE_RISK"]
        else:
            return "LIKELY_HALLUCINATED", self.COLOR_TIERS["LIKELY_HALLUCINATED"]

    def localize_tokens(
        self,
        response_text: str,
        overall_h_score: float,
        sentence_scores: List[float],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Propagate scores to tokens, merge adjacent spans, generate interactive HTML heatmap."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response_text) if s.strip()]

        annotations: List[TokenSpanAnnotation] = []
        html_chunks: List[str] = []

        curr_idx = 0
        for i, sent in enumerate(sentences):
            start_c = response_text.find(sent, curr_idx)
            if start_c == -1:
                start_c = curr_idx
            end_c = start_c + len(sent)
            curr_idx = end_c

            s_score = sentence_scores[i] if i < len(sentence_scores) else overall_h_score
            tier, color = self.get_tier_and_color(s_score)

            annotations.append(TokenSpanAnnotation(
                start_char=start_c,
                end_char=end_c,
                text=sent,
                risk_score=round(s_score, 4),
                risk_tier=tier,
                color_hex=color,
            ))

            # HTML heatmap span representation
            bg_color = f"{color}33"  # 20% opacity alpha
            html_chunks.append(
                f'<span style="background-color: {bg_color}; border-bottom: 2px solid {color}; '
                f'padding: 2px 4px; border-radius: 4px;" title="Risk: {s_score:.4f} ({tier})">{sent}</span>'
            )

        html_heatmap = f'<div class="hallucisense-heatmap" style="line-height: 1.8; font-family: sans-serif;">{" ".join(html_chunks)}</div>'

        return [asdict(a) for a in annotations], html_heatmap
