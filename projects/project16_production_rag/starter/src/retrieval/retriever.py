"""
TODO — Implement HybridRetriever combining BM25 (keyword) + vector (semantic) search.

Hybrid search architecture:
  ┌──────────┐    ┌─────────────┐
  │  BM25    │──► │             │
  │(keyword) │    │  RRF Fusion │──► ranked list
  │          │    │             │
  │  Vector  │──► │             │
  │(semantic)│    └─────────────┘
  └──────────┘

Reciprocal Rank Fusion (RRF) formula:
  score(d) = Σ  1 / (k + rank_i(d))
  where k = 60 (constant that dampens high-rank advantage)

Why hybrid?
  - BM25 is great for exact keyword matches ("API rate limit")
  - Vector search is great for semantic similarity ("how many calls per hour")
  - Together they handle both

Dependencies:
  from rank_bm25 import BM25Okapi
  from src.ingestion.embedder import embed_query
  from src.store.chroma_store import VectorStore
"""
from __future__ import annotations
import logging
from rank_bm25 import BM25Okapi
from src.config import cfg
from src.models import RetrievedChunk, RetrievalResult
from src.store.chroma_store import VectorStore
from src.ingestion.embedder import embed_query

logger = logging.getLogger(__name__)
_RRF_K = 60


class HybridRetriever:

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[dict] = []
        self.rebuild_bm25()   # build index from existing chunks at startup

    def rebuild_bm25(self) -> None:
        """
        Load all chunks from ChromaDB and build a fresh BM25Okapi index.

        TODO 1: Call self._store.get_all_documents() → list of dicts
        TODO 2: Store result in self._bm25_chunks
        TODO 3: Tokenize each chunk: tokens = text.lower().split()
        TODO 4: Build BM25Okapi(tokenized_corpus) and store in self._bm25
        TODO 5: Log how many docs are in the index
        """
        raise NotImplementedError

    def retrieve(
        self,
        query: str,
        top_k: int = cfg.TOP_K,
        mode: str = "hybrid",   # "vector" | "bm25" | "hybrid"
    ) -> RetrievalResult:
        """
        Main retrieval entry point.

        TODO 6: Route to _vector_search, _bm25_search, or _hybrid_search based on mode
        TODO 7: Wrap raw results into RetrievalResult(chunks=..., mode=mode, query=query)
        """
        raise NotImplementedError

    def _vector_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        TODO 8: embed_query(query) → embedding vector
        TODO 9: self._store.query(embedding, n_results=top_k) → raw results
        TODO 10: Convert each raw dict to RetrievedChunk(
                    chunk_id=r["chunk_id"], text=r["text"],
                    source=r["metadata"]["source"],
                    score=1 - r["distance"],   # cosine similarity
                 )
        """
        raise NotImplementedError

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        TODO 11: Tokenize query: tokens = query.lower().split()
        TODO 12: scores = self._bm25.get_scores(tokens)
        TODO 13: Sort indices by score descending, take top_k
        TODO 14: Convert to list[RetrievedChunk], normalise score to [0,1]
        """
        raise NotImplementedError

    def _hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        Fuse vector + BM25 results using Reciprocal Rank Fusion.

        TODO 15: Get vector_results = _vector_search(query, top_k)
        TODO 16: Get bm25_results  = _bm25_search(query, top_k)
        TODO 17: Build scores dict: chunk_id → RRF score
                   For vector rank i: scores[id] += 1 / (_RRF_K + i + 1)
                   For bm25   rank i: scores[id] += 1 / (_RRF_K + i + 1)
        TODO 18: Sort by total RRF score desc, return top_k RetrievedChunks
        """
        raise NotImplementedError
