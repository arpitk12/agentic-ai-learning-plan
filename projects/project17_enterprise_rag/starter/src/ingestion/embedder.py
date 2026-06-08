"""Sentence-transformers embedding wrapper (given complete)."""
from __future__ import annotations
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from src.config import cfg

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = cfg.embedding_model) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, batch_size=cfg.embedding_batch_size, normalize_embeddings=True, show_progress_bar=False).tolist()
