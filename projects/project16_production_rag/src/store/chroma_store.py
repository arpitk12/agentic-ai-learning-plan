"""
ChromaDB abstraction — works in local (persistent) mode or HTTP mode.
All other modules interact with the vector store through this interface only.
"""
from __future__ import annotations
import logging
from typing import Optional
import chromadb
from chromadb.config import Settings
from src.config import cfg

logger = logging.getLogger(__name__)


def _make_client() -> chromadb.ClientAPI:
    """Return a ChromaDB client — HTTP if CHROMA_HOST is set, else local persistent."""
    if cfg.CHROMA_HOST:
        logger.info("ChromaDB HTTP mode: %s:%s", cfg.CHROMA_HOST, cfg.CHROMA_PORT)
        return chromadb.HttpClient(host=cfg.CHROMA_HOST, port=cfg.CHROMA_PORT)
    cfg.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("ChromaDB local mode: %s", cfg.CHROMA_DIR)
    return chromadb.PersistentClient(
        path=str(cfg.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


class VectorStore:
    """
    Thin wrapper around a single ChromaDB collection.

    Usage:
        store = VectorStore()
        store.upsert(ids, embeddings, documents, metadatas)
        results = store.query(query_embedding, n_results=6)
        store.delete(ids)
    """

    def __init__(self, collection_name: Optional[str] = None):
        self._client     = _make_client()
        self._name       = collection_name or cfg.COLLECTION_NAME
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info("Collection '%s' ready — %d docs", self._name, self.count())

    # ── Write operations (ingestion pipeline) ──────────────────────────────

    def upsert(
        self,
        ids:        list[str],
        embeddings: list[list[float]],
        documents:  list[str],
        metadatas:  list[dict],
    ) -> int:
        """Insert or update chunks. Returns number of items written."""
        if not ids:
            return 0
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def delete(self, ids: list[str]):
        """Delete chunks by ID (e.g. when a document is updated)."""
        if ids:
            self._collection.delete(ids=ids)

    def delete_by_doc(self, doc_id: str):
        """Delete all chunks belonging to a document."""
        results = self._collection.get(where={"doc_id": {"$eq": doc_id}})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            logger.info("Deleted %d chunks for doc '%s'", len(results["ids"]), doc_id)

    # ── Read operations (serving pipeline) ─────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 6,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Vector similarity search. Returns raw ChromaDB result dict:
        {ids, distances, documents, metadatas}
        """
        kw: dict = dict(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kw["where"] = where
        return self._collection.query(**kw)

    def get_all_documents(self) -> dict:
        """Return all stored documents (for BM25 index rebuild)."""
        return self._collection.get(include=["documents", "metadatas"])

    def count(self) -> int:
        return self._collection.count()

    def stats(self) -> dict:
        return {
            "collection":    self._name,
            "total_chunks":  self.count(),
            "chroma_mode":   "http" if cfg.CHROMA_HOST else "local",
        }
