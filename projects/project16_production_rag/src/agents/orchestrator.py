"""
Orchestrator — classifies intent and routes each query to the correct agent.

Intent categories:
  rag     — questions about documents, product features, policies, how-to
  direct  — greetings, meta questions, math, general knowledge (no doc needed)

The orchestrator is the single entry point for all queries from routes.py.
Adding a new agent = add a branch here + a new file in src/agents/.
"""
from __future__ import annotations
import logging
from src.config import cfg
from src.models import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

_INTENT_SYSTEM = (
    "Classify this message into exactly one category.\n"
    "  rag    — questions about specific documents, product features, policies, pricing, how-to\n"
    "  direct — greetings, meta questions, math, general knowledge not needing documents\n"
    "Reply with ONLY one word."
)


async def classify_intent(question: str) -> str:
    import litellm
    resp = litellm.completion(
        model=cfg.MODEL,
        messages=[
            {"role": "system",  "content": _INTENT_SYSTEM},
            {"role": "user",    "content": question},
        ],
        max_tokens=5,
        temperature=0,
    )
    intent = resp.choices[0].message.content.strip().lower()
    return intent if intent in ("rag", "direct") else "rag"   # default to RAG


async def handle(
    request:    QueryRequest,
    retriever,                  # HybridRetriever — avoids circular import
    request_id: str = "",
) -> QueryResponse:
    """Classify intent → dispatch to the appropriate agent."""
    from src.agents.rag_agent    import rag_agent
    from src.agents.direct_agent import direct_agent

    intent = await classify_intent(request.question)
    logger.info("[%s] intent=%s  question='%s'", request_id[:8], intent, request.question[:60])

    if intent == "direct":
        return await direct_agent(request, request_id=request_id)
    return await rag_agent(request, retriever, request_id=request_id)
