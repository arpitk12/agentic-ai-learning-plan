"""Sentence-transformers embedding wrapper with batch support."""
from __future__ import annotations

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from src.config import cfg

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = cfg.embedding_model) -> None:
        logger.info("Loading embedding model '%s'...", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model ready (dim=%d)", self._model.get_sentence_embedding_dimension())

    def embed_text(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=cfg.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
