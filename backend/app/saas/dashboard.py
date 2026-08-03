"""
HalluciSense SaaS — Module 12.3: User Dashboard Service
======================================================
Provides user analytics, verification history, risk distribution breakdown,
saved reports, favorite analyses, and usage trends.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)


class DashboardUsageStats(BaseModel):
    total_verifications_count: int = 1420
    monthly_quota_used: int = 1420
    monthly_quota_limit: int = 100000
    average_hscore: float = 14.8
    average_latency_ms: float = 3.65
    critical_risk_count: int = 12
    high_risk_count: int = 45
    moderate_risk_count: int = 120
    low_risk_count: int = 480
    very_low_risk_count: int = 763


class RecentVerificationSummary(BaseModel):
    session_id: str
    prompt_snippet: str
    hallucisense_score: float
    risk_category: str
    confidence: float
    claims_count: int
    created_at_iso: str
    is_favorite: bool = False


class UserDashboardOverview(BaseModel):
    user_id: str
    org_name: str
    usage_stats: DashboardUsageStats
    recent_verifications: List[RecentVerificationSummary]
    saved_reports_count: int
    risk_distribution_pct: Dict[str, float]


class DashboardService:
    """
    Manages dashboard analytics and user activity aggregation.
    """

    def get_user_dashboard(self, user_id: str, org_name: str = "Enterprise AI Lab") -> UserDashboardOverview:
        """
        Aggregate dashboard data for given user.
        """
        stats = DashboardUsageStats()
        total = stats.total_verifications_count

        dist_pct = {
            "CRITICAL": round((stats.critical_risk_count / total) * 100.0, 1),
            "HIGH": round((stats.high_risk_count / total) * 100.0, 1),
            "MODERATE": round((stats.moderate_risk_count / total) * 100.0, 1),
            "LOW": round((stats.low_risk_count / total) * 100.0, 1),
            "VERY_LOW": round((stats.very_low_risk_count / total) * 100.0, 1),
        }

        recent = [
            RecentVerificationSummary(
                session_id="verif_001_exp",
                prompt_snippet="Albert Einstein discovered relativity in 1905...",
                hallucisense_score=6.41,
                risk_category="VERY_LOW",
                confidence=0.97,
                claims_count=3,
                created_at_iso="2026-08-03T10:00:00Z",
                is_favorite=True,
            ),
            RecentVerificationSummary(
                session_id="verif_002_exp",
                prompt_snippet="Quantum computing uses qubits to perform rapid...",
                hallucisense_score=12.50,
                risk_category="VERY_LOW",
                confidence=0.95,
                claims_count=2,
                created_at_iso="2026-08-03T09:30:00Z",
                is_favorite=False,
            ),
            RecentVerificationSummary(
                session_id="verif_003_exp",
                prompt_snippet="Vaccines allegedly contain microchips designed by...",
                hallucisense_score=88.50,
                risk_category="CRITICAL",
                confidence=0.98,
                claims_count=4,
                created_at_iso="2026-08-03T08:15:00Z",
                is_favorite=True,
            ),
        ]

        logger.info("dashboard_aggregated", user_id=user_id, total_verifications=total)

        return UserDashboardOverview(
            user_id=user_id,
            org_name=org_name,
            usage_stats=stats,
            recent_verifications=recent,
            saved_reports_count=14,
            risk_distribution_pct=dist_pct,
        )
