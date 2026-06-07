"""
Direct agent — answers questions using only the LLM, no retrieval.
Used for conversational / meta queries that don't need document context.
"""
from __future__ import annotations
import logging
import time
from src.config import cfg
from src.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

_DIRECT_SYSTEM = (
    "You are a helpful, concise assistant. Answer the question clearly. "
    "If you are not sure, say so — do not guess."
)


async def direct_agent(request: QueryRequest, request_id: str = "") -> QueryResponse:
    import litellm
    t0 = time.perf_counter()
    resp = litellm.completion(
        model=cfg.MODEL,
        messages=[
            {"role": "system", "content": _DIRECT_SYSTEM},
            {"role": "user",   "content": request.question},
        ],
        max_tokens=cfg.LLM_MAX_TOKENS,
        temperature=cfg.LLM_TEMPERATURE,
    )
    return QueryResponse(
        answer=resp.choices[0].message.content.strip(),
        citations=[], agent="direct",
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        request_id=request_id, model=cfg.MODEL,
    )
