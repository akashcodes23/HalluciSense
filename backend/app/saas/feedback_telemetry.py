"""
HalluciSense SaaS — Module 13.4: User Feedback & Opt-In Telemetry
==================================================================
Manages in-app user feedback submissions, bug reports, feature requests,
satisfaction ratings, and opt-in anonymous operational telemetry collection.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)


class UserFeedbackItem(BaseModel):
    feedback_id: str
    feedback_type: str = "BUG_REPORT"  # 'BUG_REPORT', 'FEATURE_REQUEST', 'SATISFACTION'
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: str
    user_email: Optional[str] = None
    created_at_iso: str


class TelemetryEvent(BaseModel):
    event_id: str
    session_id: str
    event_type: str  # 'VERIFICATION_EXECUTED', 'REPORT_EXPORTED'
    latency_ms: float
    opt_in_consent: bool = True
    created_at_iso: str


class FeedbackTelemetryService:
    """
    Service recording user feedback submissions and anonymous telemetry.
    """

    def __init__(self):
        self._feedback_store: List[UserFeedbackItem] = []
        self._telemetry_store: List[TelemetryEvent] = []

    def submit_feedback(
        self, feedback_type: str, comment: str, rating: Optional[int] = None, email: Optional[str] = None
    ) -> UserFeedbackItem:
        """Record user feedback submission."""
        item = UserFeedbackItem(
            feedback_id=f"fb_{uuid.uuid4().hex[:8]}",
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            user_email=email,
            created_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._feedback_store.append(item)
        logger.info("user_feedback_submitted", feedback_id=item.feedback_id, feedback_type=feedback_type)
        return item

    def record_telemetry(self, event_type: str, latency_ms: float, opt_in: bool = True) -> Optional[TelemetryEvent]:
        """Record anonymous opt-in operational telemetry."""
        if not opt_in:
            return None

        event = TelemetryEvent(
            event_id=f"tel_{uuid.uuid4().hex[:8]}",
            session_id=f"sess_{uuid.uuid4().hex[:6]}",
            event_type=event_type,
            latency_ms=latency_ms,
            opt_in_consent=True,
            created_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._telemetry_store.append(event)
        logger.debug("telemetry_event_recorded", event_id=event.event_id)
        return event
