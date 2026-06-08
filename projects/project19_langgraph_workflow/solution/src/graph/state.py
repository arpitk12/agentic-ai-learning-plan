"""LangGraph state definition."""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class ReviewState(TypedDict):
    # Append-only via operator.add reducer — nodes add messages, never replace
    messages: Annotated[list, operator.add]

    # Last-write-wins fields
    code: str
    language: str
    parsed_info: dict                   # functions, classes, line count
    security_issues: list[str]
    quality_score: float                # 0-10
    review: str                         # LLM-generated review
    human_feedback: Optional[str]       # set during human_approval node
    revision_count: int                 # track how many revisions happened
    status: str  # parsed|analyzed|review_ready|approved|needs_revision|finalized
