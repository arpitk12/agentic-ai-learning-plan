"""
Embed consumer — reads chunks from 'document-chunks', embeds them in batches,
publishes to 'embedded-chunks'.

Pipeline: document-chunks → [embed_consumer] → embedded-chunks

Key optimisations
─────────────────
  Batch embedding: accumulate 32 chunks, embed in one forward pass (10× faster
  than embedding one at a time).

  Embedding cache: before embedding, check Redis for a cached embedding keyed
  by sha256(text). Cache hit rate in practice: 20-40% (duplicate chunks from
  shared boilerplate, headers, etc.).

  Poll timeout 50ms: allows batch to fill up while not blocking too long.
"""
from __future__ import annotations

import json
import logging
import signal

from confluent_kafka import Consumer, KafkaError, Producer

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
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": self.GROUP_ID,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
                "max.poll.interval.ms": 300_000,
            }
        )
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "acks": "all",
                "compression.type": "snappy",
            }
        )
        self._embedder = Embedder()
        self._cache = RedisCache(url=redis_url)
        self._batch_size = cfg.kafka_embed_batch_size
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def run(self) -> None:
        self._consumer.subscribe([cfg.kafka_chunks_topic])
        logger.info(
            "EmbedConsumer started — batch_size=%d", self._batch_size
        )

        batch: list[dict] = []

        while self._running:
            msg = self._consumer.poll(timeout=0.05)  # 50ms — lets batch fill

            if msg and not msg.error():
                batch.append(json.loads(msg.value()))
            elif msg and msg.error().code() != KafkaError._PARTITION_EOF:
                logger.error("Kafka error: %s", msg.error())

            # Flush batch when full or when nothing arrived (pipeline drain)
            should_flush = len(batch) >= self._batch_size or (batch and msg is None)
            if not should_flush:
                continue

            self._process_batch(batch)
            batch.clear()

        self._consumer.close()

    def _process_batch(self, batch: list[dict]) -> None:
        texts = [item["text"] for item in batch]

        # Split into cached vs to-embed
        embeddings: list[list[float]] = []
        to_embed_indices: list[int] = []
        to_embed_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get_embedding(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # placeholder
                to_embed_indices.append(i)
                to_embed_texts.append(text)

        # Batch embed cache misses
        if to_embed_texts:
            new_embeddings = self._embedder.embed_batch(to_embed_texts)
            for idx, emb, text in zip(to_embed_indices, new_embeddings, to_embed_texts):
                embeddings[idx] = emb
                self._cache.set_embedding(text, emb)

        # Publish to embedded-chunks
        for item, emb in zip(batch, embeddings):
            item["embedding"] = emb
            self._producer.produce(
                topic=cfg.kafka_embedded_topic,
                key=item["doc_id"].encode(),
                value=json.dumps(item).encode(),
            )

        self._producer.flush()
        self._consumer.commit()
        logger.debug("Embedded and published %d chunks", len(batch))

    def _handle_shutdown(self, *_) -> None:
        self._running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    EmbedConsumer().run()
