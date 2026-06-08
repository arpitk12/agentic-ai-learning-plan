"""ComplianceState — LangGraph TypedDict with audit reducer."""
from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class ComplianceState(TypedDict):
    """State for the enterprise compliance review workflow.

    Fields marked Annotated[list, operator.add] use LangGraph's
    append-only reducer — safe for concurrent node writes.
    """
    # ── Document ────────────────────────────────────────────────────────
    document_id: str
    document_content: str
    document_type: str            # contract | policy | data-processing | vendor

    # ── Analysis results ─────────────────────────────────────────────────
    risk_assessment: dict[str, Any]    # RiskAssessment.model_dump()
    policy_check: dict[str, Any]       # PolicyCheckResult.model_dump()
    compliance_score: float

    # ── Human review ─────────────────────────────────────────────────────
    human_feedback: str | None
    human_reviewer: str | None
    escalated_to_legal: bool

    # ── Append-only records (reducer: list concatenation) ─────────────────
    audit_entries: Annotated[list, operator.add]
    messages: Annotated[list, operator.add]

    # ── Cost tracking ─────────────────────────────────────────────────────
    llm_cost_usd: float
    processing_time_seconds: float

    # ── Status ───────────────────────────────────────────────────────────
    status: str   # intake|risk_assessed|policy_checked|approved|rejected|escalated
