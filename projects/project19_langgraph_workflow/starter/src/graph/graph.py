"""Starter stub — Project 19: Build the LangGraph StateGraph."""
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
    """Compile the code review StateGraph.

    Node order:
        START → parse_code → analyze_security → analyze_quality
              → generate_review → human_approval → (conditional) → finalize_review → END
                                                  ↘ (needs_revision) → generate_review
    """
    # TODO 1: Create builder = StateGraph(ReviewState)
    # TODO 2: Add all 6 nodes with builder.add_node("name", function)
    # TODO 3: Add linear edges: START→parse_code→analyze_security→analyze_quality→generate_review→human_approval
    # TODO 4: Add finalize_review → END
    # TODO 5: Add conditional edges from "human_approval" using route_approval:
    #         "approved" → "finalize_review", "needs_revision" → "generate_review"
    # TODO 6: Return builder.compile(checkpointer=checkpointer, interrupt_before=["human_approval"])
    raise NotImplementedError
