"""
LLM-based reranker.

After retrieval returns the top-K candidates, the reranker asks the LLM to score
each chunk's relevance to the query on a 0–10 scale. The top-N highest-scoring
chunks are kept for generation. Falls back to the original retrieval order if
the LLM call fails.
"""
from __future__ import annotations
import logging
from litellm import completion
from src.config import cfg
from src.models import RetrievedChunk

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = cfg.RERANK_TOP_N,
) -> list[RetrievedChunk]:
    """
    Rerank chunks using the LLM and return the top top_n.

    TODO 1: Return early if chunks is empty or already small enough to skip reranking
    TODO 2: Attempt LLM reranking; on any exception fall back to the original order
    """
    raise NotImplementedError


def _llm_rerank(
    query: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Score each chunk with the LLM and return sorted by score descending.

    TODO 3: For each chunk, ask the LLM to rate how relevant the chunk is to the query
    TODO 4: Parse the integer score from the response, defaulting to 5 on parse errors
    TODO 5: Return the chunks sorted by score descending
    """
    raise NotImplementedError
