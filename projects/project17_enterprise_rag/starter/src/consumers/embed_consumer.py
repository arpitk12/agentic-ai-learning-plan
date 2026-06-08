"""
Embed consumer — reads document-chunks, embeds in batches, publishes to embedded-chunks.
Key optimisation: batch 32 chunks per forward pass + embedding cache.
See GUIDE.md Phase 2.4 Step 4.
"""
from __future__ import annotations

import json
import logging
import signal

from src.cache.redis_cache import RedisCache
from src.config import cfg
from src.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


class EmbedConsumer:
    GROUP_ID = "embedder-group"

    def __init__(
        self,
        bootstrap_servers: str = cfg.kafka_bootstrap_servers,
        redis_url: str = cfg.redis_url,
    ) -> None:
        """
        TODO 1: Create Kafka Consumer (group=embedder-group, enable.auto.commit=False)
                and Producer.
                Create self._embedder = Embedder().
                Create self._cache = RedisCache(url=redis_url).
                Set self._batch_size = cfg.kafka_embed_batch_size (32).
                Set self._running = True.
        """
        raise NotImplementedError

    def run(self) -> None:
        """
        TODO 2: Subscribe to cfg.kafka_chunks_topic.
                Initialise batch = [].

        TODO 3: Poll loop (timeout=0.05 — 50ms for batch accumulation):
                  if msg is valid: append parsed JSON to batch
                  if len(batch) >= self._batch_size OR (batch and msg is None):
                    self._process_batch(batch)
                    batch.clear()
        """
        raise NotImplementedError

    def _process_batch(self, batch: list[dict]) -> None:
        """
        TODO 4: For each item in batch, check self._cache.get_embedding(item["text"]).
                Separate into cached and to-embed groups.

        TODO 5: Batch-embed the cache misses:
                  self._embedder.embed_batch(to_embed_texts)
                  Store each result in Redis: self._cache.set_embedding(text, emb).

        TODO 6: Publish each item (with "embedding" field added) to cfg.kafka_embedded_topic.

        TODO 7: Call self._producer.flush() and then self._consumer.commit().
        """
        raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    EmbedConsumer().run()
