"""
RAG agent — retrieves relevant chunks then generates a grounded answer.
The answer must cite sources; hallucination is reduced by instruction.
"""
from __future__ import annotations
import json
import logging
import time
from src.config import cfg
from src.models import Citation, QueryRequest, QueryResponse, RetrievedChunk
from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

_RAG_SYSTEM = """You are a precise, helpful assistant. Answer the user's question
using ONLY the provided context passages. Follow these rules:
1. Base every claim on the context — do not introduce outside knowledge.
2. If the context does not contain the answer, say "I don't have that information."
3. Be concise (under 200 words unless the question requires detail).
4. Do not repeat the question."""


async def rag_agent(
    request:   QueryRequest,
    retriever: HybridRetriever,
    request_id: str = "",
) -> QueryResponse:
    """Retrieve relevant chunks, rerank, then generate a grounded answer."""
    import litellm

    t0 = time.perf_counter()

    # 1. Retrieve
    top_k  = request.top_k or cfg.TOP_K
    result = retriever.retrieve(request.question, top_k=top_k, mode=request.mode)

    if not result.chunks:
        return QueryResponse(
            answer="I couldn't find any relevant information. Please try rephrasing your question.",
            citations=[], agent="rag", retrieval=result,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            request_id=request_id, model=cfg.MODEL,
        )

    # 2. Rerank
    chunks = await rerank(request.question, result.chunks) if request.rerank else result.chunks[:cfg.RERANK_TOP_N]
    result.chunks = chunks

    # 3. Build context string
    context = "\n\n---\n\n".join(
        f"[Source: {c.title}]\n{c.content}" for c in chunks
    )

    # 4. Generate
    messages = [
        {"role": "system", "content": _RAG_SYSTEM},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {request.question}"},
    ]
    resp   = litellm.completion(
        model=cfg.MODEL,
        messages=messages,
        max_tokens=cfg.LLM_MAX_TOKENS,
        temperature=cfg.LLM_TEMPERATURE,
    )
    answer = resp.choices[0].message.content.strip()

    # 5. Build citations
    citations = [
        Citation(
            title=c.title, source=c.source,
            chunk_id=c.chunk_id, snippet=c.content[:150],
        )
        for c in chunks
    ]

    return QueryResponse(
        answer=answer, citations=citations, agent="rag",
        retrieval=result,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        request_id=request_id, model=cfg.MODEL,
    )
