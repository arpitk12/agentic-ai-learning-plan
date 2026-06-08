"""FastAPI routes for the LangGraph code review workflow."""
from __future__ import annotations

import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.graph.checkpointer import get_checkpointer
from src.graph.graph import build_graph

router = APIRouter(prefix="/review", tags=["code-review"])


class ReviewRequest(BaseModel):
    code: str
    language: str = "python"
    thread_id: str | None = None


class FeedbackRequest(BaseModel):
    thread_id: str
    feedback: str  # "approve" or "revise: <notes>"


@router.post("/start")
async def start_review(req: ReviewRequest):
    """Submit code for review. Returns thread_id for follow-up."""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state_input = {"code": req.code, "language": req.language, "messages": []}

    with get_checkpointer() as cp:
        graph = build_graph(checkpointer=cp)
        result = await graph.ainvoke(state_input, config=config)

    return {
        "thread_id": thread_id,
        "status": result.get("status"),
        "review": result.get("review"),
        "security_issues": result.get("security_issues", []),
        "quality_score": result.get("quality_score"),
    }


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit human feedback to resume an interrupted review."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": req.thread_id}}
    with get_checkpointer() as cp:
        graph = build_graph(checkpointer=cp)
        result = await graph.ainvoke(Command(resume=req.feedback), config=config)

    return {
        "thread_id": req.thread_id,
        "status": result.get("status"),
        "review": result.get("review"),
    }


@router.get("/stream/{thread_id}")
async def stream_review(thread_id: str, code: str, language: str = "python"):
    """Stream review updates via Server-Sent Events."""
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        with get_checkpointer() as cp:
            graph = build_graph(checkpointer=cp)
            async for event in graph.astream_events(
                {"code": code, "language": language, "messages": []},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chain_end":
                    node = event.get("name", "")
                    data = event.get("data", {}).get("output", {})
                    yield f"data: node={node} status={data.get('status', '')}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
