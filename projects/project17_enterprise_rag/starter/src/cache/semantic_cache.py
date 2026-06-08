"""
Semantic cache — returns a cached answer when the new question is
semantically similar (cosine ≥ threshold) to a previously answered one.

See GUIDE.md Phase 3 for the full design rationale.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

import numpy as np

from src.config import cfg

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(
        self,
        threshold: float = cfg.semantic_cache_threshold,
        max_size: int = cfg.semantic_cache_size,
    ) -> None:
        """
        TODO 1: Store threshold and max_size as instance attributes.
                Initialise self._store as an OrderedDict (preserves insertion order for LRU).
                Initialise self._hits = 0 and self._misses = 0.
        """
        raise NotImplementedError

    def get(self, q_emb: np.ndarray) -> Optional[str]:
        """
        TODO 2: Normalise q_emb with _normalise().

        TODO 3: Iterate over self._store.items().
                For each (emb_bytes, answer_json):
                  - Reconstruct the cached embedding: np.frombuffer(emb_bytes, dtype=np.float32)
                  - Compute cosine similarity: float(np.dot(q_norm, cached))
                  - If sim >= self._threshold:
                      * Move entry to end (LRU): self._store.move_to_end(emb_bytes)
                      * Increment self._hits
                      * Return answer_json

        TODO 4: If no hit, increment self._misses and return None.
        """
        raise NotImplementedError

    def set(self, q_emb: np.ndarray, answer_json: str) -> None:
        """
        TODO 5: Normalise q_emb and convert to bytes (key = q_norm.astype(np.float32).tobytes()).
                Add to self._store.
                If len(self._store) > self._max_size, evict the oldest entry
                with self._store.popitem(last=False).
        """
        raise NotImplementedError

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """TODO 6: Return self._hits / (self._hits + self._misses) or 0.0."""
        raise NotImplementedError

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        v = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
