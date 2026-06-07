"""
Singleton sentence-transformers embedder.

Loads the embedding model once on first use. Both the offline ingestion script
and the live API share the same model via this module, keeping all vectors
in the same embedding space.

The model is configured by EMBED_MODEL (default: all-MiniLM-L6-v2, 384 dimensions).
Embeddings are L2-normalised so cosine similarity equals the dot product.
"""
from __future__ import annotations
from sentence_transformers import SentenceTransformer
from src.config import cfg

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    TODO 1: Load the SentenceTransformer model on first call and cache it.
            Return the cached model on subsequent calls.
    """
    raise NotImplementedError


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    TODO 2: Encode a batch of strings with normalised embeddings.
    TODO 3: Return the result as a plain Python list of lists (not numpy arrays).
    """
    raise NotImplementedError


def embed_query(query: str) -> list[float]:
    """TODO 4: Embed a single string and return a single vector."""
    raise NotImplementedError
