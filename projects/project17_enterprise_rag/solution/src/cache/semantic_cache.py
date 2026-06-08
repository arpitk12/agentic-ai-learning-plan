"""
Semantic cache — returns a cached answer when the new question is
semantically similar (cosine ≥ threshold) to a previously answered question.

Design
──────
  In-memory LRU dict:  {embedding_bytes: answer_json}
  Max size: SEMANTIC_CACHE_SIZE entries (default 5 000).
  Comparison: O(N) cosine similarity scan in numpy.

  For multi-replica deployments the cache is per-process.
  Production upgrade: store embeddings in Redis Stack and use the
  VSIM command for server-side approximate nearest-neighbour search.

Why 0.97 threshold?
──────────────────
  Cosine similarity of 0.97 between two sentence embeddings (384-dim)
  corresponds to near-paraphrases.  E.g.:
    "What is the API rate limit?"  ↔  "How many API calls per minute?"  ≈ 0.98
    "What is the API rate limit?"  ↔  "How do I reset my password?"     ≈ 0.21
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
        self._threshold = threshold
        self._max_size = max_size
        # OrderedDict preserves insertion order → O(1) LRU eviction
        self._store: OrderedDict[bytes, str] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ── Public interface ──────────────────────────────────────────────────

    def get(self, q_emb: np.ndarray) -> Optional[str]:
        """
        Return cached answer if any stored embedding is within threshold.
        Returns None on miss.
        """
        q_norm = self._normalise(q_emb)
        for emb_bytes, answer_json in self._store.items():
            cached = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = float(np.dot(q_norm, cached))
            if sim >= self._threshold:
                # Move to end (most recently used)
                self._store.move_to_end(emb_bytes)
                self._hits += 1
                logger.debug("Semantic cache hit (sim=%.4f)", sim)
                return answer_json
        self._misses += 1
        return None

    def set(self, q_emb: np.ndarray, answer_json: str) -> None:
        """Store embedding + answer; evict oldest entry if at capacity."""
        q_norm = self._normalise(q_emb)
        key = q_norm.tobytes()
        if key in self._store:
            self._store.move_to_end(key)
        else:
            self._store[key] = answer_json
            if len(self._store) > self._max_size:
                # Evict the least recently used entry
                self._store.popitem(last=False)

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        v = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
