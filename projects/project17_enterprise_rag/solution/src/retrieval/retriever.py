"""
Hybrid retriever — combines Qdrant vector search with BM25 keyword search,
fused via Reciprocal Rank Fusion (RRF).

Why hybrid?
──────────
  Vector search: high semantic recall — finds "cancel subscription" when query says "terminate plan"
  BM25 search:   high keyword precision — reliably finds "error code 404" or "v2.3.1 changelog"
  RRF fusion:    neither individually is as robust as both combined

RRF formula
───────────
  score(chunk) = Σ_i  1 / (k + rank_i)
  k=60 is the standard constant (dampens the effect of very high ranks).

  A chunk ranked #1 in both lists: 1/61 + 1/61 = 0.0328
  A chunk ranked #1 in one, absent in other: 1/61 = 0.0164
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.config import cfg
from src.ingestion.embedder import Embedder
from src.store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

_RRF_K = 60


class HybridRetriever:
    def __init__(
        self,
        qdrant_store: QdrantStore,
        embedder: Embedder,
        top_k: int = cfg.retrieval_top_k,
    ) -> None:
        self._store = qdrant_store
        self._embedder = embedder
        self._top_k = top_k
        # BM25 index is rebuilt lazily on first query (or explicitly via rebuild_bm25)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[dict] = []   # [{chunk_id, text, title, source}]

    def search(
        self,
        question: str,
        top_k: int | None = None,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        top_k = top_k or self._top_k
        q_emb = self._embedder.embed_text(question)

        vector_results = self._vector_search(q_emb, top_k, source_filter)
        bm25_results = self._bm25_search(question, top_k)

        fused = self._rrf_fuse(vector_results, bm25_results, top_k)
        logger.debug(
            "Hybrid search '%s...' → %d results (vector=%d, bm25=%d)",
            question[:50], len(fused), len(vector_results), len(bm25_results),
        )
        return fused

    # ── Retrieval modes ───────────────────────────────────────────────────

    def _vector_search(
        self, q_emb: List[float], top_k: int, source_filter: Optional[str]
    ) -> List[dict]:
        return self._store.search(
            query_vector=q_emb,
            top_k=top_k,
            source_filter=source_filter,
        )

    def _bm25_search(self, question: str, top_k: int) -> List[dict]:
        if self._bm25 is None or not self._bm25_corpus:
            logger.debug("BM25 index not built — skipping BM25 leg")
            return []

        tokens = question.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked_indices[:top_k]:
            chunk = self._bm25_corpus[i].copy()
            chunk["score"] = float(scores[i])
            results.append(chunk)
        return results

    def _rrf_fuse(
        self,
        vector_results: List[dict],
        bm25_results: List[dict],
        top_k: int,
    ) -> List[dict]:
        rrf_scores: dict[str, float] = defaultdict(float)
        all_chunks: dict[str, dict] = {}

        for rank, item in enumerate(vector_results):
            cid = item["chunk_id"]
            rrf_scores[cid] += 1.0 / (_RRF_K + rank + 1)
            all_chunks[cid] = item

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            rrf_scores[cid] += 1.0 / (_RRF_K + rank + 1)
            if cid not in all_chunks:
                all_chunks[cid] = item

        ranked = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
        return [
            {**all_chunks[cid], "score": rrf_scores[cid]}
            for cid in ranked[:top_k]
        ]

    # ── BM25 index ────────────────────────────────────────────────────────

    def rebuild_bm25(self, corpus: Optional[List[dict]] = None) -> None:
        """
        (Re)build the BM25 index from a corpus of chunks.

        In a production system, this corpus comes from a snapshot of the
        Qdrant collection, fetched periodically or on stale-flag trigger.

        Args:
            corpus: List of {chunk_id, text, title, source} dicts.
                    If None, uses the existing self._bm25_corpus.
        """
        if corpus is not None:
            self._bm25_corpus = corpus

        if not self._bm25_corpus:
            logger.warning("BM25 rebuild: corpus is empty")
            return

        tokenised = [item["text"].lower().split() for item in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index rebuilt — %d documents", len(self._bm25_corpus))
