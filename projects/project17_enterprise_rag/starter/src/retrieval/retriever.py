"""
Hybrid retriever — Qdrant vector search + BM25 + RRF fusion.
See GUIDE.md Phase 5.1 for the RRF formula.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Optional

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
        """
        TODO 1: Store qdrant_store, embedder, top_k as instance attributes.
                Initialise self._bm25 = None and self._bm25_corpus = [].
        """
        raise NotImplementedError

    def search(
        self,
        question: str,
        top_k: int | None = None,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        TODO 2: Compute q_emb = self._embedder.embed_text(question).
        TODO 3: Run _vector_search(q_emb, top_k, source_filter).
        TODO 4: Run _bm25_search(question, top_k).
        TODO 5: Fuse results with _rrf_fuse() and return top_k results.
        """
        raise NotImplementedError

    def _vector_search(
        self, q_emb: List[float], top_k: int, source_filter: Optional[str]
    ) -> List[dict]:
        """TODO 6: Delegate to self._store.search()."""
        raise NotImplementedError

    def _bm25_search(self, question: str, top_k: int) -> List[dict]:
        """
        TODO 7: If self._bm25 is None, return [].
                Tokenise question.lower().split().
                Get scores from self._bm25.get_scores(tokens).
                Sort indices by score descending, return top_k items from self._bm25_corpus.
        Hint: from rank_bm25 import BM25Okapi
        """
        raise NotImplementedError

    def _rrf_fuse(
        self,
        vector_results: List[dict],
        bm25_results: List[dict],
        top_k: int,
    ) -> List[dict]:
        """
        TODO 8: For each list, score each chunk as 1/(60 + rank + 1) and add to a defaultdict.
                Collect all chunks in a dict keyed by chunk_id.
                Sort chunk_ids by RRF score descending and return top_k.
        """
        raise NotImplementedError

    def rebuild_bm25(self, corpus: Optional[List[dict]] = None) -> None:
        """
        TODO 9: If corpus is provided, update self._bm25_corpus.
                Tokenise each item's text field: item["text"].lower().split().
                Build self._bm25 = BM25Okapi(tokenised_corpus).
        """
        raise NotImplementedError
