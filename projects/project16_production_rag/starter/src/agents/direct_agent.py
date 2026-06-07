"""
TODO — Implement the direct LLM agent (no retrieval, no context).

Used when the orchestrator decides the question doesn't need the knowledge base
(e.g. "What is 2+2?", "Write me a haiku", "Explain JSON in general").

This is intentionally simple — just a raw LLM call with a helpful system prompt.
"""
from __future__ import annotations
import logging
from litellm import completion
from src.config import cfg
from src.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a helpful, knowledgeable assistant. "
    "Answer questions clearly and concisely."
)


async def direct_agent(
    request: QueryRequest,
    request_id: str = "?",
) -> QueryResponse:
    """
    Answer a question with a direct LLM call (no retrieval).

    TODO 1: Call litellm.completion(
                model=cfg.MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": request.question},
                ],
            )
    TODO 2: Extract answer = response.choices[0].message.content
    TODO 3: Return QueryResponse(answer=answer, model=cfg.MODEL,
                                 citations=[],
                                 retrieval_mode="none",
                                 request_id=request_id)
    """
    raise NotImplementedError
