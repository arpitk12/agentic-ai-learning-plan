"""
TODO — Implement the multi-agent orchestrator.

The orchestrator:
  1. Classifies the incoming question as "rag" or "direct" using a fast LLM call
  2. Routes to rag_agent (uses knowledge base) or direct_agent (raw LLM)

Classification prompt:
  "Does this question require looking up specific product/API documentation?
   Reply with exactly one word: 'rag' or 'direct'.
   Question: {question}"

Why classify?
  - Not every question needs retrieval ("What's the capital of France?")
  - RAG adds latency + cost; only use it when it helps
  - The classifier uses max_tokens=5 so it's nearly free
"""
from __future__ import annotations
import logging
from litellm import completion
from src.config import cfg
from src.models import QueryRequest, QueryResponse
from src.retrieval.retriever import HybridRetriever
from src.agents.rag_agent import rag_agent
from src.agents.direct_agent import direct_agent

logger = logging.getLogger(__name__)


def classify_intent(question: str) -> str:
    """
    Ask the LLM whether the question needs retrieval.
    Returns "rag" or "direct".

    TODO 1: Build the classification prompt (see module docstring)
    TODO 2: Call litellm.completion(model=cfg.MODEL, messages=[...], max_tokens=5, temperature=0)
    TODO 3: Extract the response text, strip/lower it
    TODO 4: Return "rag" if "rag" in response else "direct"
    TODO 5: Wrap in try/except — on any error, default to "rag"
    """
    raise NotImplementedError


async def handle(
    request: QueryRequest,
    retriever: HybridRetriever,
    request_id: str = "?",
) -> QueryResponse:
    """
    Classify intent and dispatch to the right agent.

    TODO 6: Call intent = classify_intent(request.question)
    TODO 7: Log the classification decision
    TODO 8: If intent == "rag": return await rag_agent(request, retriever, request_id)
    TODO 9: Else:               return await direct_agent(request, request_id)
    """
    raise NotImplementedError
