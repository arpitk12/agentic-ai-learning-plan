"""
TODO — Implement the VectorStore wrapper around ChromaDB.

This module is the *only* place in the codebase that talks to ChromaDB.
All other modules (ingestion, retrieval) call this class.

Design decisions to implement:
  - Support TWO client modes controlled by env var CHROMA_HOST:
      * empty  → chromadb.PersistentClient(path=cfg.CHROMA_PATH)   (local dev)
      * set    → chromadb.HttpClient(host, port=8001)               (Docker)
  - Single shared collection named cfg.COLLECTION_NAME
  - All vectors are stored with metadata: doc_id, chunk_id, title, source, index

Reference: https://docs.trychroma.com/
"""
from __future__ import annotations
from typing import Any
import chromadb
from src.config import cfg


class VectorStore:
    """ChromaDB-backed vector store with local/HTTP dual mode."""

    def __init__(self) -> None:
        # TODO 1: Call self._make_client() to create self._client
        # TODO 2: Call self._client.get_or_create_collection(
        #             name=cfg.COLLECTION_NAME,
        #             metadata={"hnsw:space": "cosine"},
        #         ) and store as self._col
        raise NotImplementedError

    def _make_client(self) -> chromadb.Client:
        """
        TODO 3: Return the right ChromaDB client depending on cfg.CHROMA_HOST.

        if cfg.CHROMA_HOST:
            return chromadb.HttpClient(host=cfg.CHROMA_HOST, port=8001)
        else:
            return chromadb.PersistentClient(path=cfg.CHROMA_PATH)
        """
        raise NotImplementedError

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        TODO 4: Call self._col.upsert(ids=ids, embeddings=embeddings,
                                      documents=documents, metadatas=metadatas)
        """
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        """TODO 5: Call self._col.delete(ids=ids)"""
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> None:
        """
        TODO 6: Delete all chunks that belong to a document.
        Hint: self._col.delete(where={"doc_id": doc_id})
        """
        raise NotImplementedError

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        TODO 7: Query ChromaDB and return a flat list of dicts.

        Steps:
          raw = self._col.query(
              query_embeddings=[query_embedding],
              n_results=n_results,
              include=["documents", "metadatas", "distances"],
          )
          Then zip raw["ids"][0], raw["documents"][0],
               raw["metadatas"][0], raw["distances"][0]
          and return list of dicts:
            {"chunk_id": ..., "text": ..., "metadata": ..., "distance": ...}
        """
        raise NotImplementedError

    def get_all_documents(self) -> list[dict[str, Any]]:
        """
        TODO 8: Return all stored chunks (used to rebuild BM25 index).
        Hint: self._col.get(include=["documents", "metadatas"])
        Then zip ids, documents, metadatas into list of dicts.
        """
        raise NotImplementedError

    def count(self) -> int:
        """TODO 9: Return self._col.count()"""
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        """
        TODO 10: Return a dict with at least:
          {"collection": cfg.COLLECTION_NAME, "chunks": self.count(),
           "embedding_model": cfg.EMBED_MODEL}
        """
        raise NotImplementedError
