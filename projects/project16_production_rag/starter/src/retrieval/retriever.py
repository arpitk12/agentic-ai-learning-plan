"""
HybridRetriever combines BM25 keyword search with vector semantic search,
then fuses the two ranked lists using Reciprocal Rank Fusion (RRF).

RRF score for a document:  sum of  1 / (k + rank_i)  across all lists,  where k = 60.
A larger constant k reduces the advantage of top-ranked positions.
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
        self.rebuild_bm25()

    def rebuild_bm25(self) -> None:
        """
        Rebuild the BM25 index from all chunks currently in ChromaDB.

        TODO 1: Fetch all documents from the store
        TODO 2: Tokenise each document's text into lowercase words
        TODO 3: Build a new BM25 index from the tokenised corpus
        TODO 4: Log the number of documents now in the index
        """
        raise NotImplementedError

    def retrieve(
        self,
        query: str,
        top_k: int = cfg.TOP_K,
        mode: str = "hybrid",
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        TODO 5: Dispatch to the appropriate search method based on mode ("vector", "bm25", or "hybrid")
        TODO 6: Return a RetrievalResult wrapping the chunks, mode, and query
        """
        raise NotImplementedError

    def _vector_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        TODO 7: Embed the query into a vector
        TODO 8: Query the vector store for the nearest neighbours
        TODO 9: Convert raw results into RetrievedChunk objects with a normalised relevance score
        """
        raise NotImplementedError

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        TODO 10: Tokenise the query into lowercase words
        TODO 11: Score all indexed documents with BM25
        TODO 12: Return the top-k as RetrievedChunk objects with normalised scores
        """
        raise NotImplementedError

    def _hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        Fuse vector and BM25 results using Reciprocal Rank Fusion.

        TODO 13: Run both vector search and BM25 search
        TODO 14: Accumulate an RRF score for each document based on its rank in each list
        TODO 15: Sort by total RRF score descending and return the top-k chunks
        """
        raise NotImplementedError
