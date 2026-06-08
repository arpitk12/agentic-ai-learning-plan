"""Starter stub — Project 23: LangGraph State."""
from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


class ComplianceState(TypedDict):
    """State for the compliance review workflow.

    Annotated[list, operator.add] = append-only (LangGraph reducer).
    Never overwrite audit_entries — only append to them.
    """
    # TODO 1: document_id: str
    # TODO 2: document_content: str
    # TODO 3: document_type: str
    # TODO 4: risk_assessment: dict[str, Any]
    # TODO 5: policy_check: dict[str, Any]
    # TODO 6: compliance_score: float
    # TODO 7: human_feedback: str | None
    # TODO 8: human_reviewer: str | None
    # TODO 9: escalated_to_legal: bool
    # TODO 10: audit_entries: Annotated[list, operator.add]
    # TODO 11: messages: Annotated[list, operator.add]
    # TODO 12: llm_cost_usd: float
    # TODO 13: processing_time_seconds: float
    # TODO 14: status: str
    pass
