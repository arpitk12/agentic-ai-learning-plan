"""
LLM-based reranker — takes the top-K retrieved chunks and reorders them
by relevance to the query, returning only the top-N most relevant.

Why rerank? Vector similarity finds semantically close content, but
the highest-scoring chunk is not always the most useful for generation.
Reranking adds a targeted relevance signal before passing context to the LLM.
"""
from __future__ import annotations
import json
import logging
from src.config import cfg
from src.models import RetrievedChunk

logger = logging.getLogger(__name__)


async def rerank(
    query:  str,
    chunks: list[RetrievedChunk],
    top_n:  int | None = None,
) -> list[RetrievedChunk]:
    """
    Score each chunk for relevance to the query using the LLM, then return top_n.
    Falls back to original order if LLM call fails.
    """
    n = top_n or cfg.RERANK_TOP_N
    if len(chunks) <= n:
        return chunks   # nothing to rerank

    try:
        return await _llm_rerank(query, chunks, n)
    except Exception as exc:
        logger.warning("Reranker failed (%s) — using original order", exc)
        return chunks[:n]


async def _llm_rerank(
    query: str, chunks: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    """Score chunks 0–1 with an LLM, sort descending, return top_n."""
    import litellm
    from src.config import cfg as _cfg

    # Build a compact prompt with all candidate chunks
    candidates = "\n\n".join(
        f"[{i}] (source: {c.title})\n{c.content[:300]}"
        for i, c in enumerate(chunks)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Rank each passage 0.0–1.0 for relevance to the query.\n\n"
        f"{candidates}\n\n"
        f"Return ONLY valid JSON: {{\"scores\": [score_0, score_1, ...]}}\n"
        f"One float per passage in the same order. No explanation."
    )
    resp  = litellm.completion(
        model=_cfg.MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0,
    )
    raw   = resp.choices[0].message.content.strip()
    raw   = raw.removeprefix("```json").removesuffix("```").strip()
    data  = json.loads(raw)
    scores: list[float] = data.get("scores", [])

    if len(scores) != len(chunks):
        raise ValueError(f"Score count mismatch: got {len(scores)}, expected {len(chunks)}")

    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = round(float(score), 4)

    ranked = sorted(chunks, key=lambda c: c.rerank_score or 0, reverse=True)
    return ranked[:top_n]
