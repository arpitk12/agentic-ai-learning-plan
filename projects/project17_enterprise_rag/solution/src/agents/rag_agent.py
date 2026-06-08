"""
Full RAG agent — wires together retrieval → reranking → generation → faithfulness → citations.

Every query goes through:
  1. Semantic cache check    (< 5 ms on hit)
  2. Hybrid retrieval        (Qdrant + BM25 + RRF)
  3. Cross-encoder reranker  (top-20 → top-5)
  4. Abstain gate 1          (retrieval score < threshold)
  5. LLM generation          (grounded prompt)
  6. Faithfulness check      (NLI per-sentence)
  7. Abstain gate 2          (overall faithfulness < threshold)
  8. Citation verification
  9. Cache store + return
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import List, Optional

import litellm
import numpy as np
from sentence_transformers import CrossEncoder

from src.cache.redis_cache import RedisCache
from src.cache.semantic_cache import SemanticCache
from src.config import cfg
from src.hallucination.abstain_policy import AbstainPolicy
from src.hallucination.citation_verifier import CitationVerifier
from src.hallucination.faithfulness_checker import FaithfulnessChecker
from src.ingestion.embedder import Embedder
from src.models import QueryRequest, QueryResponse
from src.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_GROUNDED_PROMPT = """\
You are a precise assistant. Answer the question using ONLY the information \
in the provided context. Do not add any information that is not explicitly stated \
in the context. If the context does not contain sufficient information to answer \
the question, write exactly:
"I cannot answer this based on the available documentation."

Context:
{context}

Question: {question}

Answer:"""


class RAGAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        embedder: Embedder,
        faithfulness_checker: FaithfulnessChecker,
        reranker_model: CrossEncoder,
        redis_cache: RedisCache,
        semantic_cache: SemanticCache,
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        self._checker = faithfulness_checker
        self._reranker = reranker_model
        self._redis = redis_cache
        self._sem_cache = semantic_cache
        self._abstain_policy = AbstainPolicy()
        self._citation_verifier = CitationVerifier(nli_model=faithfulness_checker._model)

    def answer(self, request: QueryRequest, request_id: str = "") -> QueryResponse:
        request_id = request_id or str(uuid.uuid4())
        t0 = time.perf_counter()

        # ── Step 1: Exact query cache ──────────────────────────────────────
        cached_raw = self._redis.get_query(request.question)
        if cached_raw is not None:
            resp = QueryResponse(**cached_raw, cached=True, request_id=request_id)
            resp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return resp

        # ── Step 2: Semantic cache ─────────────────────────────────────────
        q_emb = np.array(self._embedder.embed_text(request.question))
        sem_hit = self._sem_cache.get(q_emb)
        if sem_hit is not None:
            resp = QueryResponse(**json.loads(sem_hit), cached=True, request_id=request_id)
            resp.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return resp

        # ── Step 3: Hybrid retrieval ───────────────────────────────────────
        chunks = self._retriever.search(
            request.question,
            top_k=cfg.retrieval_top_k,
            source_filter=request.source_filter,
        )
        retrieval_max_score = max((c["score"] for c in chunks), default=0.0)

        # ── Step 4: Cross-encoder rerank top-20 → top-5 ───────────────────
        if chunks:
            pairs = [(request.question, c["text"]) for c in chunks]
            rerank_scores = self._reranker.predict(pairs)
            ranked = sorted(zip(rerank_scores, chunks), key=lambda x: x[0], reverse=True)
            chunks = [c for _, c in ranked[:cfg.reranker_top_n]]

        # ── Step 5: Abstain gate 1 — no relevant docs ─────────────────────
        temp_faith = _empty_faithfulness()
        should_abstain, reason = self._abstain_policy.should_abstain(
            retrieval_max_score, temp_faith
        )
        # Only check retrieval score at this stage
        if retrieval_max_score < cfg.min_retrieval_score:
            return self._abstain_response(
                request_id=request_id,
                reason="no_relevant_documents",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # ── Step 6: LLM generation ────────────────────────────────────────
        context = "\n\n".join(
            f"[Source {i + 1}: {c['title']}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        prompt = _GROUNDED_PROMPT.format(context=context, question=request.question)
        try:
            completion = litellm.completion(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,   # deterministic — critical for faithfulness
            )
            raw_answer = completion.choices[0].message.content or ""
        except Exception as exc:
            logger.exception("LLM generation failed: %s", exc)
            return self._abstain_response(request_id, "llm_error", (time.perf_counter() - t0) * 1000)

        # ── Step 7: Faithfulness check ────────────────────────────────────
        faith_result = self._checker.check(raw_answer, chunks)

        # ── Step 8: Abstain gate 2 — low faithfulness ─────────────────────
        should_abstain, reason = self._abstain_policy.should_abstain(
            retrieval_max_score, faith_result
        )
        if should_abstain:
            return self._abstain_response(request_id, reason, (time.perf_counter() - t0) * 1000)

        # ── Step 9: Citation verification ─────────────────────────────────
        citations = self._citation_verifier.verify(faith_result.grounded_answer, chunks)

        # ── Step 10: Cache and return ──────────────────────────────────────
        response = QueryResponse(
            answer=faith_result.grounded_answer,
            citations=citations,
            faithfulness_score=faith_result.faithfulness_score,
            retrieval_score=retrieval_max_score,
            abstained=False,
            cached=False,
            request_id=request_id,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        resp_dict = response.model_dump()
        self._redis.set_query(request.question, resp_dict)
        self._sem_cache.set(q_emb, json.dumps(resp_dict))

        return response

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _abstain_response(request_id: str, reason: str, latency_ms: float) -> QueryResponse:
        return QueryResponse(
            answer="",
            abstained=True,
            abstain_reason=reason,
            request_id=request_id,
            latency_ms=round(latency_ms, 1),
        )


def _empty_faithfulness():
    from src.models import FaithfulnessResult
    return FaithfulnessResult(sentences=[], faithfulness_score=1.0, passed=True, grounded_answer="x")
