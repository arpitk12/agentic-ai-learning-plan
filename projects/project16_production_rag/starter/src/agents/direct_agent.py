"""
Direct LLM agent — answers without retrieval.

Used for general questions that do not require the knowledge base.
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

    TODO 1: Call the LLM with the system prompt and the user's question
    TODO 2: Extract the answer text from the response
    TODO 3: Return a QueryResponse with no citations and retrieval_mode set to "none"
    """
    raise NotImplementedError
