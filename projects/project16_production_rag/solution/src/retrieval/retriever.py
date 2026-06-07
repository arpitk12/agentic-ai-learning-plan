"""
Hybrid retriever — combines BM25 keyword search with ChromaDB vector search
using Reciprocal Rank Fusion (RRF). Each method captures different signal;
hybrid consistently outperforms either alone.

Architecture:
  query → [vector search] + [BM25 search]
              ↓                   ↓
         vector ranks        BM25 ranks
              └────── RRF fusion ──────┘
                          ↓
                   sorted candidates
"""
from __future__ import annotations
import logging
import time
from rank_bm25 import BM25Okapi
from src.config import cfg
from src.models import RetrievedChunk, RetrievalResult
from src.store.chroma_store import VectorStore
from src.ingestion.embedder import embed_query

logger = logging.getLogger(__name__)

_RRF_K = 60   # RRF constant — standard value from the literature


def _rrf(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


class HybridRetriever:
    """
    Retriever that maintains a BM25 index over all stored chunks (in memory)
    and combines it with ChromaDB vector search.

    BM25 index is rebuilt when `rebuild_bm25()` is called — call it after ingestion.
    At serve time, the index is already in memory so retrieval is fast.
    """

    def __init__(self, store: VectorStore | None = None):
        self._store    = store or VectorStore()
        self._bm25:    BM25Okapi | None = None
        self._docs:    list[str] = []
        self._ids:     list[str] = []
        self._metas:   list[dict] = []
        self.rebuild_bm25()

    # ── BM25 index management ──────────────────────────────────────────────

    def rebuild_bm25(self):
        """Rebuild the in-memory BM25 index from the current ChromaDB contents."""
        all_docs = self._store.get_all_documents()
        self._docs   = all_docs.get("documents", []) or []
        self._ids    = all_docs.get("ids", [])        or []
        self._metas  = all_docs.get("metadatas", [])  or []

        if self._docs:
            tokenised    = [d.lower().split() for d in self._docs]
            self._bm25   = BM25Okapi(tokenised)
            logger.info("BM25 index built: %d documents", len(self._docs))
        else:
            self._bm25 = None
            logger.warning("BM25 index empty — no documents in store yet")

    # ── Core retrieval ─────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",    # vector | bm25 | hybrid
    ) -> RetrievalResult:
        """
        Retrieve the top_k most relevant chunks for a query.
        mode='hybrid' (default) fuses BM25 + vector via RRF.
        """
        k  = top_k or cfg.TOP_K
        t0 = time.perf_counter()

        if mode == "vector":
            chunks = self._vector_search(query, k)
        elif mode == "bm25":
            chunks = self._bm25_search(query, k)
        else:
            chunks = self._hybrid_search(query, k)

        return RetrievalResult(
            query=query, chunks=chunks, mode=mode,
            top_k=k, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # ── Private search methods ─────────────────────────────────────────────

    def _vector_search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._store.count() == 0:
            return []
        qvec    = embed_query(query)
        results = self._store.query(qvec, n_results=k)
        chunks: list[RetrievedChunk] = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB distance is 1 - cosine_similarity for cosine space
            score = 1 - results["distances"][0][i]
            meta  = results["metadatas"][0][i]
            chunks.append(RetrievedChunk(
                chunk_id=doc_id, doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""), title=meta.get("title", ""),
                content=results["documents"][0][i],
                vector_score=round(score, 4),
            ))
        return chunks

    def _bm25_search(self, query: str, k: int) -> list[RetrievedChunk]:
        if not self._bm25 or not self._docs:
            return []
        q_tokens = query.lower().split()
        scores   = self._bm25.get_scores(q_tokens)
        top_idx  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        chunks: list[RetrievedChunk] = []
        for idx in top_idx:
            if scores[idx] == 0:
                break
            meta = self._metas[idx] if idx < len(self._metas) else {}
            chunks.append(RetrievedChunk(
                chunk_id=self._ids[idx], doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""), title=meta.get("title", ""),
                content=self._docs[idx],
                bm25_score=round(float(scores[idx]), 4),
            ))
        return chunks

    def _hybrid_search(self, query: str, k: int) -> list[RetrievedChunk]:
        """RRF fusion of vector and BM25 rankings."""
        vec_chunks = self._vector_search(query, k)
        bm25_chunks= self._bm25_search(query, k)

        # Build RRF score map: chunk_id → rrf_score
        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, RetrievedChunk] = {}

        for rank, c in enumerate(vec_chunks, 1):
            rrf_scores[c.chunk_id] = rrf_scores.get(c.chunk_id, 0) + cfg.VECTOR_WEIGHT * _rrf(rank)
            chunk_map[c.chunk_id]  = c

        for rank, c in enumerate(bm25_chunks, 1):
            rrf_scores[c.chunk_id] = rrf_scores.get(c.chunk_id, 0) + cfg.BM25_WEIGHT * _rrf(rank)
            if c.chunk_id not in chunk_map:
                chunk_map[c.chunk_id] = c
            else:
                # Merge BM25 score into existing chunk
                chunk_map[c.chunk_id].bm25_score = c.bm25_score

        # Sort by RRF and return top-k
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:k]
        result: list[RetrievedChunk] = []
        for cid in sorted_ids:
            c = chunk_map[cid]
            c.rrf_score = round(rrf_scores[cid], 4)
            result.append(c)
        return result
