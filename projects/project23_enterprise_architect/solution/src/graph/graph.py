"""Build and compile the compliance review StateGraph."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.config import cfg
from src.graph.nodes import (
    audit_log_node,
    auto_approve_node,
    human_review_node,
    intake_node,
    policy_check_node,
    report_node,
    risk_analysis_node,
    route_by_risk,
)
from src.graph.state import ComplianceState


def build_compliance_graph(checkpointer=None):
    """Compile the compliance review workflow.

    Flow:
        START
          → intake
          → risk_analysis
          → policy_check
          → [conditional] high/critical or non-compliant → human_review
                          low/medium and compliant       → auto_approve
          → audit_log
          → report
          → END

    interrupt_before=["human_review"] pauses graph execution for HITL.
    Resume with: graph.invoke(Command(resume="approve: <notes>"), config=config)
    """
    builder = StateGraph(ComplianceState)

    builder.add_node("intake",        intake_node)
    builder.add_node("risk_analysis", risk_analysis_node)
    builder.add_node("policy_check",  policy_check_node)
    builder.add_node("human_review",  human_review_node)
    builder.add_node("auto_approve",  auto_approve_node)
    builder.add_node("audit_log",     audit_log_node)
    builder.add_node("report",        report_node)

    builder.add_edge(START,           "intake")
    builder.add_edge("intake",        "risk_analysis")
    builder.add_edge("risk_analysis", "policy_check")

    builder.add_conditional_edges(
        "policy_check",
        route_by_risk,
        {"human_review": "human_review", "auto_approve": "auto_approve"},
    )

    builder.add_edge("human_review", "audit_log")
    builder.add_edge("auto_approve", "audit_log")
    builder.add_edge("audit_log",    "report")
    builder.add_edge("report",       END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )


@contextmanager
def get_graph():
    """Context manager yielding a compiled graph with SqliteSaver checkpointer."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path(cfg.checkpoint_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(db_path)) as cp:
        yield build_compliance_graph(checkpointer=cp)
