"""
VectorStore wraps ChromaDB — the only place in the codebase that touches ChromaDB directly.
All other modules call this class.

Supports two client modes selected by the CHROMA_HOST env var:
  - Empty: local persistent storage (development)
  - Set:   HTTP client connecting to a remote ChromaDB service (Docker / production)

All chunks are stored with metadata: doc_id, chunk_id, title, source, index.
Reference: https://docs.trychroma.com/
"""
from __future__ import annotations
from typing import Any
import chromadb
from src.config import cfg


class VectorStore:
    """ChromaDB-backed vector store with local/HTTP dual mode."""

    def __init__(self) -> None:
        # TODO 1: Create the ChromaDB client by calling _make_client()
        # TODO 2: Create or open a collection named cfg.COLLECTION_NAME using cosine distance
        raise NotImplementedError

    def _make_client(self) -> chromadb.Client:
        """
        TODO 3: Return a local persistent client if cfg.CHROMA_HOST is empty,
                or an HTTP client pointing at cfg.CHROMA_HOST otherwise.
        """
        raise NotImplementedError

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """TODO 4: Write chunks into the collection (insert or overwrite by id)."""
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        """TODO 5: Delete chunks by their ids."""
        raise NotImplementedError

    def delete_by_doc(self, doc_id: str) -> None:
        """TODO 6: Delete all chunks that belong to the given document."""
        raise NotImplementedError

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        TODO 7: Query the collection with a query embedding and return the top n_results.
                Return a flat list of dicts, each with keys: chunk_id, text, metadata, distance.
        """
        raise NotImplementedError

    def get_all_documents(self) -> list[dict[str, Any]]:
        """
        TODO 8: Return every stored chunk as a list of dicts.
                Used by the retriever to rebuild the BM25 index at startup.
        """
        raise NotImplementedError

    def count(self) -> int:
        """TODO 9: Return the total number of chunks in the collection."""
        raise NotImplementedError

    def stats(self) -> dict[str, Any]:
        """
        TODO 10: Return a summary dict with at least the collection name,
                 chunk count, and embedding model name.
        """
        raise NotImplementedError
