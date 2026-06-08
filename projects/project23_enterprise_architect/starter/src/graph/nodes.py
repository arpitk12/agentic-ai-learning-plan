"""Starter stub — Project 23: LangGraph nodes."""
from __future__ import annotations

from langgraph.types import interrupt
from src.graph.state import ComplianceState


async def intake_node(state: ComplianceState) -> dict:
    """TODO 1: Validate document exists and return {document_content, status, audit_entries}"""
    raise NotImplementedError


async def risk_analysis_node(state: ComplianceState) -> dict:
    """
    TODO 2: Call assess_document_risk(state["document_content"], state["document_type"])
    TODO 3: Return {risk_assessment: assessment.model_dump(), llm_cost_usd, status, audit_entries}
    """
    raise NotImplementedError


async def policy_check_node(state: ComplianceState) -> dict:
    """
    TODO 4: Load policies (from cfg or MCP)
    TODO 5: Call check_document_policies(...)
    TODO 6: Return {policy_check, compliance_score, llm_cost_usd, status, audit_entries}
    """
    raise NotImplementedError


async def human_review_node(state: ComplianceState) -> dict:
    """
    TODO 7: Call interrupt({document_id, risk_level, prompt: "approve/reject/escalate: <notes>"})
    TODO 8: Parse action from returned feedback string
    TODO 9: Return {human_feedback, human_reviewer, escalated_to_legal, status, audit_entries}
    """
    raise NotImplementedError


async def auto_approve_node(state: ComplianceState) -> dict:
    """TODO 10: Return {status: "approved", audit_entries: [...]}"""
    raise NotImplementedError


async def audit_log_node(state: ComplianceState) -> dict:
    """TODO 11: Write audit summary and return updated audit_entries"""
    raise NotImplementedError


async def report_node(state: ComplianceState) -> dict:
    """TODO 12: Calculate cost savings vs manual, return final audit_entries with savings data"""
    raise NotImplementedError


def route_by_risk(state: ComplianceState) -> str:
    """
    TODO 13: Check risk_level and is_compliant
    Return "human_review" for high/critical or non-compliant
    Return "auto_approve" otherwise
    """
    raise NotImplementedError
