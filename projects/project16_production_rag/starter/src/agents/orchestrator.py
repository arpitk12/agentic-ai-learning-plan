"""
Multi-agent orchestrator.

Classifies each incoming question as needing retrieval ("rag") or not ("direct"),
then routes to the appropriate agent. The classifier uses a minimal LLM call
so the routing cost is negligible.
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
    Ask the LLM whether this question requires looking up product or API documentation.
    Returns "rag" or "direct".

    TODO 1: Build a prompt that asks whether the question needs documentation lookup
    TODO 2: Call the LLM with a very small token budget (classification only, not generation)
    TODO 3: Parse the response and return "rag" or "direct"
    TODO 4: Default to "rag" on any exception
    """
    raise NotImplementedError


async def handle(
    request: QueryRequest,
    retriever: HybridRetriever,
    request_id: str = "?",
) -> QueryResponse:
    """
    Classify intent and dispatch to the right agent.

    TODO 5: Classify the intent of the question
    TODO 6: Log the routing decision
    TODO 7: Route to the RAG agent or the direct agent based on the result
    """
    raise NotImplementedError
