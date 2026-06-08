"""Starter stub — Project 19: LangGraph State definition."""
from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class ReviewState(TypedDict):
    """State for the code review workflow.

    Annotated[list, operator.add] is a LangGraph 'reducer':
    it APPENDS new messages instead of replacing them.
    """
    # TODO 1: Add 'messages' field as Annotated[list, operator.add]
    # TODO 2: Add 'code' field (str)  — the code being reviewed
    # TODO 3: Add 'language' field (str) — programming language
    # TODO 4: Add 'parsed_info' field (dict[str, Any]) — AST analysis results
    # TODO 5: Add 'security_issues' field (list[str]) — OWASP check results
    # TODO 6: Add 'quality_score' field (float) — 0-10 quality rating
    # TODO 7: Add 'review' field (str) — generated review text
    # TODO 8: Add 'human_feedback' field (str | None) — reviewer notes
    # TODO 9: Add 'revision_count' field (int) — how many revisions done
    # TODO 10: Add 'status' field (str) — workflow status
    pass  # Remove this after implementing
