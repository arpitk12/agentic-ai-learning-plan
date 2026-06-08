"""
Index consumer — reads embedded-chunks, upserts to Qdrant in batches of 256.
See GUIDE.md Phase 2.4 Step 5.
"""
from __future__ import annotations

import json
import logging
import signal

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
        """
        TODO 1: Create Kafka Consumer (group=indexer-group, enable.auto.commit=False,
                max.poll.interval.ms=600000 — large batches take time).
                Create self._store = QdrantStore(url=qdrant_url).
                Call self._store.create_collection() (no-op if exists).
                Set self._batch_size = cfg.kafka_index_batch_size (256).
                Set self._running = True.
        """
        raise NotImplementedError

    def run(self) -> None:
        """
        TODO 2: Subscribe to cfg.kafka_embedded_topic.
                Initialise batch = [].

        TODO 3: Poll loop (timeout=0.05):
                  if valid msg: append parsed JSON to batch
                  if len(batch) >= self._batch_size OR (batch and msg is None):
                    try:
                      self._store.upsert_batch(batch)
                      self._consumer.commit()
                    except Exception:
                      log error — do NOT commit (Kafka redelivers the batch)
                    finally:
                      batch.clear()

        Note: NOT committing on Qdrant failure is the key at-least-once guarantee.
              Qdrant upserts are idempotent, so redelivery is safe.
        """
        raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    IndexConsumer().run()
