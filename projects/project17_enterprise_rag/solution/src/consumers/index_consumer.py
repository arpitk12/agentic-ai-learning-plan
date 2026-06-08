"""
Index consumer — reads embedded chunks from 'embedded-chunks',
upserts them to Qdrant in batches of 256.

Pipeline: embedded-chunks → [index_consumer] → Qdrant

Batch size 256
──────────────
  Qdrant upsert throughput peaks at 200-500 vectors per call.
  Larger batches reduce gRPC round-trip overhead.
  256 is a good default; tune based on vector size and network latency.

Idempotency
───────────
  Qdrant upsert uses deterministic UUIDs (uuid5 from chunk_id).
  Duplicate messages (Kafka at-least-once) produce no side effects.
"""
from __future__ import annotations

import json
import logging
import signal

from confluent_kafka import Consumer, KafkaError

from src.config import cfg
from src.store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class IndexConsumer:
    GROUP_ID = "indexer-group"

    def __init__(
        self,
        bootstrap_servers: str = cfg.kafka_bootstrap_servers,
        qdrant_url: str = cfg.qdrant_url,
    ) -> None:
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": self.GROUP_ID,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
                "max.poll.interval.ms": 600_000,  # large batches can take time
            }
        )
        self._store = QdrantStore(url=qdrant_url)
        self._store.create_collection()   # no-op if already exists
        self._batch_size = cfg.kafka_index_batch_size
        self._running = True
        self._total_indexed = 0
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def run(self) -> None:
        self._consumer.subscribe([cfg.kafka_embedded_topic])
        logger.info(
            "IndexConsumer started — batch_size=%d → Qdrant %s",
            self._batch_size, cfg.qdrant_url,
        )

        batch: list[dict] = []

        while self._running:
            msg = self._consumer.poll(timeout=0.05)

            if msg and not msg.error():
                batch.append(json.loads(msg.value()))
            elif msg and msg.error().code() != KafkaError._PARTITION_EOF:
                logger.error("Kafka error: %s", msg.error())

            should_flush = len(batch) >= self._batch_size or (batch and msg is None)
            if not should_flush:
                continue

            try:
                n = self._store.upsert_batch(batch)
                self._total_indexed += n
                self._consumer.commit()
                logger.info(
                    "Indexed %d vectors (total: %d)",
                    n, self._total_indexed,
                )
            except Exception as exc:
                logger.exception("Qdrant upsert failed: %s", exc)
                # Do NOT commit — Kafka will redeliver this batch
            finally:
                batch.clear()

        self._consumer.close()
        logger.info("IndexConsumer shut down — total indexed: %d", self._total_indexed)

    def _handle_shutdown(self, *_) -> None:
        self._running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    IndexConsumer().run()
