"""
Redis cache — two layers:

  1. Exact query cache (Redis String)
     Key:   qcache:{sha256(normalised_question)[:16]}
     Value: JSON of QueryResponse
     TTL:   REDIS_QUERY_CACHE_TTL  (default 1 h)

  2. Embedding cache (Redis String)
     Key:   emb:{sha256(text)[:16]}
     Value: JSON list[float]
     TTL:   REDIS_EMBED_CACHE_TTL  (default 24 h)
     Used by embed_consumer to skip re-embedding identical text.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

import redis as _redis

from src.config import cfg

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, url: str = cfg.redis_url) -> None:
        self._client = _redis.Redis.from_url(url, decode_responses=True)

    # ── Exact query cache ─────────────────────────────────────────────────

    @staticmethod
    def _query_key(question: str) -> str:
        normalised = question.strip().lower()
        digest = hashlib.sha256(normalised.encode()).hexdigest()[:16]
        return f"qcache:{digest}"

    def get_query(self, question: str) -> Optional[dict]:
        key = self._query_key(question)
        raw = self._client.get(key)
        if raw is None:
            return None
        logger.debug("Exact cache hit: %s", key)
        return json.loads(raw)

    def set_query(self, question: str, response: dict) -> None:
        key = self._query_key(question)
        self._client.setex(key, cfg.redis_query_cache_ttl, json.dumps(response))

    # ── Embedding cache ───────────────────────────────────────────────────

    @staticmethod
    def _embed_key(text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"emb:{digest}"

    def get_embedding(self, text: str) -> Optional[list]:
        key = self._embed_key(text)
        raw = self._client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set_embedding(self, text: str, embedding: list) -> None:
        key = self._embed_key(text)
        self._client.setex(key, cfg.redis_embed_cache_ttl, json.dumps(embedding))

    # ── Ops ───────────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        try:
            return self._client.ping()
        except Exception:
            return False

    def query_cache_size(self) -> int:
        try:
            return len(self._client.keys("qcache:*"))
        except Exception:
            return -1
