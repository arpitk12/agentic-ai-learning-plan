"""
TODO — Implement the RAG agent (retrieve → rerank → generate with citations).

Flow:
  1. Call retriever.retrieve(question, top_k, mode)   → RetrievalResult
  2. Call reranker.rerank(question, chunks, top_n)    → list[RetrievedChunk]
  3. Build grounded system prompt:
       "Answer using ONLY the context below. Cite [SOURCE] after each claim."
       + numbered context chunks
  4. Call LLM with user question
  5. Parse citations from the answer text
  6. Return QueryResponse(answer, model, citations, retrieval_mode, request_id)
"""
from __future__ import annotations
import logging
from litellm import completion
from src.config import cfg
from src.models import QueryRequest, QueryResponse, Citation, RetrievedChunk
from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


async def rag_agent(
    request: QueryRequest,
    retriever: HybridRetriever,
    request_id: str = "?",
) -> QueryResponse:
    """
    Full RAG pipeline: retrieve → rerank → generate.

    TODO 1: Call retriever.retrieve(request.question, request.top_k, request.mode)
    TODO 2: Call rerank(request.question, result.chunks, cfg.RERANK_TOP_N)
    TODO 3: Build context string:
              context = "\n\n".join(
                  f"[{i+1}] (source: {c.source})\n{c.text}"
                  for i, c in enumerate(reranked)
              )
    TODO 4: Build system prompt:
              "You are a helpful assistant. Answer using ONLY the context below.
               After each factual claim, cite the source like [1] or [2].
               If the context doesn't contain the answer, say so.\n\nContext:\n" + context
    TODO 5: Call litellm.completion(model=cfg.MODEL, messages=[system, user])
    TODO 6: Extract answer = response.choices[0].message.content
    TODO 7: Build citations list from reranked chunks:
              Citation(chunk_id=c.chunk_id, text=c.text[:200],
                       source=c.source, score=c.score)
    TODO 8: Return QueryResponse(answer=answer, model=cfg.MODEL,
                                 citations=citations,
                                 retrieval_mode=result.mode,
                                 request_id=request_id)
    """
    raise NotImplementedError
