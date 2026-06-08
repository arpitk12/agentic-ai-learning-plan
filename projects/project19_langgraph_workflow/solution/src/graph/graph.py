"""Compile the LangGraph StateGraph for the code review workflow."""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from src.graph.state import ReviewState
from src.graph.nodes import (
    parse_code,
    analyze_security,
    analyze_quality,
    generate_review,
    human_approval,
    finalize_review,
    route_approval,
)


def build_graph(checkpointer=None):
    """Build and compile the code review StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver).
                      Required for human-in-the-loop / interrupt() support.

    Returns:
        Compiled LangGraph application.
    """
    builder = StateGraph(ReviewState)

    # ── Register nodes ──────────────────────────────────────────────────
    builder.add_node("parse_code", parse_code)
    builder.add_node("analyze_security", analyze_security)
    builder.add_node("analyze_quality", analyze_quality)
    builder.add_node("generate_review", generate_review)
    builder.add_node("human_approval", human_approval)
    builder.add_node("finalize_review", finalize_review)

    # ── Linear edges ────────────────────────────────────────────────────
    builder.add_edge(START, "parse_code")
    builder.add_edge("parse_code", "analyze_security")
    builder.add_edge("analyze_security", "analyze_quality")
    builder.add_edge("analyze_quality", "generate_review")
    builder.add_edge("generate_review", "human_approval")
    builder.add_edge("finalize_review", END)

    # ── Conditional edge after human_approval ───────────────────────────
    builder.add_conditional_edges(
        "human_approval",
        route_approval,
        {
            "approved": "finalize_review",
            "needs_revision": "generate_review",  # re-generate with feedback
        },
    )

    return builder.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])
