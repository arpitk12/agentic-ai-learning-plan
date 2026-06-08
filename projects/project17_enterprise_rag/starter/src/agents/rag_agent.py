"""
Full RAG agent — wires retrieval → reranking → generation → faithfulness → citations.
See GUIDE.md Phase 5.3 for the full pipeline description.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

import numpy as np

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
in the provided context. Do not add any information not in the context. \
If the context does not contain the answer, write exactly:
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
        reranker_model,
        redis_cache: RedisCache,
        semantic_cache: SemanticCache,
    ) -> None:
        """
        TODO 1: Store all injected objects as instance attributes.
                Create self._abstain_policy = AbstainPolicy().
                Create self._citation_verifier = CitationVerifier(nli_model=faithfulness_checker._model).
        """
        raise NotImplementedError

    def answer(self, request: QueryRequest, request_id: str = "") -> QueryResponse:
        """
        TODO 2: Generate request_id = str(uuid.uuid4()) if not provided.
                Record t0 = time.perf_counter().

        TODO 3: Check exact query cache — self._redis.get_query(request.question).
                If hit: return QueryResponse(**cached_raw, cached=True, request_id=...).

        TODO 4: Compute q_emb = np.array(self._embedder.embed_text(request.question)).
                Check semantic cache — self._sem_cache.get(q_emb).
                If hit: return cached response with cached=True.

        TODO 5: Run hybrid retrieval:
                  chunks = self._retriever.search(request.question, top_k=cfg.retrieval_top_k)
                  retrieval_max_score = max(c["score"] for c in chunks) if chunks else 0.0

        TODO 6: Cross-encoder rerank (top-20 → top-5):
                  pairs = [(request.question, c["text"]) for c in chunks]
                  rerank_scores = self._reranker.predict(pairs)
                  Sort by score descending, keep top cfg.reranker_top_n.

        TODO 7: Abstain gate 1 — check retrieval_max_score < cfg.min_retrieval_score.
                  Return self._abstain_response(request_id, "no_relevant_documents", latency_ms).

        TODO 8: Build context string and run LLM generation with litellm.completion():
                  model=cfg.model, temperature=0.0,
                  messages=[{"role": "user", "content": prompt}]
                  Extract raw_answer from completion.choices[0].message.content.

        TODO 9: Run faithfulness check:
                  faith_result = self._checker.check(raw_answer, chunks)

        TODO 10: Abstain gate 2 — call self._abstain_policy.should_abstain().
                   Return abstain_response if should_abstain is True.

        TODO 11: Run citation verification:
                   citations = self._citation_verifier.verify(faith_result.grounded_answer, chunks)

        TODO 12: Build QueryResponse, store in both caches, and return.
        """
        raise NotImplementedError

    @staticmethod
    def _abstain_response(request_id: str, reason: str, latency_ms: float) -> QueryResponse:
        return QueryResponse(
            answer="",
            abstained=True,
            abstain_reason=reason,
            request_id=request_id,
            latency_ms=round(latency_ms, 1),
        )
