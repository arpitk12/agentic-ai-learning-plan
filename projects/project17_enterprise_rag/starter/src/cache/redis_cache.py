"""
Redis cache — two tiers:

  1. Exact query cache  (key: qcache:{sha256[:16]},  value: JSON response)
  2. Embedding cache    (key: emb:{sha256[:16]},      value: JSON list[float])

See GUIDE.md Phase 3 for the full caching design.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from src.config import cfg

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, url: str = cfg.redis_url) -> None:
        """
        TODO 1: Import redis and create self._client using
                redis.Redis.from_url(url, decode_responses=True).
        """
        raise NotImplementedError

    # ── Exact query cache ─────────────────────────────────────────────────

    @staticmethod
    def _query_key(question: str) -> str:
        normalised = question.strip().lower()
        return "qcache:" + hashlib.sha256(normalised.encode()).hexdigest()[:16]

    def get_query(self, question: str) -> Optional[dict]:
        """
        TODO 2: Get the value for _query_key(question) from Redis.
                Return None if not found, else json.loads(value).
        """
        raise NotImplementedError

    def set_query(self, question: str, response: dict) -> None:
        """
        TODO 3: Store json.dumps(response) at _query_key(question)
                with TTL = cfg.redis_query_cache_ttl.
                Use self._client.setex(key, ttl, value).
        """
        raise NotImplementedError

    # ── Embedding cache ───────────────────────────────────────────────────

    @staticmethod
    def _embed_key(text: str) -> str:
        return "emb:" + hashlib.sha256(text.encode()).hexdigest()[:16]

    def get_embedding(self, text: str) -> Optional[list]:
        """
        TODO 4: Get the embedding for this text from Redis.
                Return None on miss, else json.loads(value).
        """
        raise NotImplementedError

    def set_embedding(self, text: str, embedding: list) -> None:
        """
        TODO 5: Store json.dumps(embedding) at _embed_key(text)
                with TTL = cfg.redis_embed_cache_ttl.
        """
        raise NotImplementedError

    # ── Ops ───────────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """
        TODO 6: Call self._client.ping() and return True.
                Return False on any exception.
        """
        raise NotImplementedError

    def query_cache_size(self) -> int:
        """
        TODO 7: Return the count of keys matching "qcache:*".
                Return -1 on exception.
        Hint: len(self._client.keys("qcache:*"))
        """
        raise NotImplementedError
