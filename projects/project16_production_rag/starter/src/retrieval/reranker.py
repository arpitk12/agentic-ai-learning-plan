"""
TODO — Implement LLM-based reranker.

Why rerank?
  Retrieval returns top-K candidates by similarity score.
  The LLM reranker reads the actual query + each chunk and scores
  relevance more accurately (but is slower — hence only top-N after retrieval).

Pattern:  top-K retrieved  →  LLM scores each  →  return top-N

Prompt template:
  "On a scale 0-10, how relevant is this passage to the query?
   Query: {query}
   Passage: {chunk}
   Reply with ONLY a single integer 0-10."

Fallback: if the LLM call fails or parsing fails, return chunks in original order.
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
    Rerank `chunks` using the LLM and return the top `top_n`.

    TODO 1: If chunks is empty or top_n >= len(chunks), return chunks[:top_n]
    TODO 2: Call _llm_rerank(query, chunks) inside a try/except
    TODO 3: On any exception, log a warning and return chunks[:top_n] (fallback)
    """
    raise NotImplementedError


def _llm_rerank(
    query: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Score each chunk with the LLM and return sorted by score (desc).

    TODO 4: For each chunk, build the prompt (see module docstring)
    TODO 5: Call litellm.completion(model=cfg.MODEL, messages=[...], max_tokens=5, temperature=0)
    TODO 6: Parse the integer score from the response content
    TODO 7: Handle parse errors gracefully (default score = 5)
    TODO 8: Sort chunks by score descending and return
    """
    raise NotImplementedError
