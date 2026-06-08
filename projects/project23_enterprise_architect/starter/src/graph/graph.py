"""Starter stub — Project 23: Build the compliance StateGraph."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from src.graph.state import ComplianceState
from src.graph.nodes import (
    intake_node, risk_analysis_node, policy_check_node,
    human_review_node, auto_approve_node, audit_log_node, report_node,
    route_by_risk,
)


def build_compliance_graph(checkpointer=None):
    """
    TODO 1: Create builder = StateGraph(ComplianceState)
    TODO 2: Add all 7 nodes: intake, risk_analysis, policy_check,
            human_review, auto_approve, audit_log, report
    TODO 3: Add linear edges:
            START → intake → risk_analysis → policy_check
    TODO 4: Add conditional edges from policy_check using route_by_risk:
            "human_review" → human_review node
            "auto_approve" → auto_approve node
    TODO 5: Both human_review and auto_approve connect to audit_log
    TODO 6: audit_log → report → END
    TODO 7: Return builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
    """
    raise NotImplementedError
