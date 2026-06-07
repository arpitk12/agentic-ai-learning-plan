"""
Embedding wrapper — sentence-transformers (local, no API cost, consistent).
The same model MUST be used for both ingestion and retrieval.
Model is loaded once as a module-level singleton.
"""
from __future__ import annotations
import logging
import time
from src.config import cfg

logger = logging.getLogger(__name__)

_model = None   # lazy-loaded singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", cfg.EMBED_MODEL)
        t0     = time.perf_counter()
        _model = SentenceTransformer(cfg.EMBED_MODEL)
        logger.info("Model loaded in %.1fs", time.perf_counter() - t0)
    return _model


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """
    Embed a list of texts in batches.
    Returns a list of float vectors (length = cfg.EMBED_DIM).
    """
    if not texts:
        return []
    model  = _get_model()
    bsize  = batch_size or cfg.EMBED_BATCH
    t0     = time.perf_counter()
    vecs   = model.encode(
        texts,
        batch_size=bsize,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,   # cosine similarity = dot product
    )
    elapsed = time.perf_counter() - t0
    logger.debug("Embedded %d texts in %.2fs (%.0f/s)", len(texts), elapsed, len(texts)/elapsed)
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Uses same model as embed_texts."""
    return embed_texts([query])[0]
