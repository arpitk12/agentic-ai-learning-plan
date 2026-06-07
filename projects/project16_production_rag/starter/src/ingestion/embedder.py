"""
TODO — Implement a singleton sentence-transformers embedder.

Key requirements:
  - Load the model ONCE (lazy singleton) — loading takes ~2s, batch calls are fast
  - Use sentence_transformers.SentenceTransformer(cfg.EMBED_MODEL)
  - Always call encode(..., normalize_embeddings=True) so cosine similarity = dot product
  - Convert numpy arrays → plain Python lists (JSON serializable)

Why the same model for ingest AND query?
  Embedding space only makes sense if both are encoded by the same model.
  cfg.EMBED_MODEL (default: "all-MiniLM-L6-v2") produces 384-dim vectors.
"""
from __future__ import annotations
from sentence_transformers import SentenceTransformer
from src.config import cfg

# Module-level singleton — populated on first call
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    TODO 1: Use the global _model variable.
    If _model is None, create SentenceTransformer(cfg.EMBED_MODEL) and store it.
    Return _model.
    """
    raise NotImplementedError


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of strings.

    TODO 2: Call _get_model().encode(texts, normalize_embeddings=True)
    TODO 3: Convert result to list[list[float]] using .tolist()
    """
    raise NotImplementedError


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    TODO 4: Reuse embed_texts([query]) and return the first (only) element.
    """
    raise NotImplementedError
