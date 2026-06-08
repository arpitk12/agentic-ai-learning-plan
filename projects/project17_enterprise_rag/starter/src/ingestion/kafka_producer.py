"""
Kafka producer — publishes raw documents to the 'raw-documents' topic.

See GUIDE.md Phase 2 for design rationale (acks, retries, compression).
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from src.config import cfg

logger = logging.getLogger(__name__)


class DocumentProducer:
    def __init__(self, bootstrap_servers: str = cfg.kafka_bootstrap_servers) -> None:
        """
        TODO 1: Import confluent_kafka.Producer.
                Create self._producer with config:
                  {
                    "bootstrap.servers": bootstrap_servers,
                    "acks": "all",          ← wait for all in-sync replicas
                    "retries": 5,
                    "retry.backoff.ms": 500,
                    "compression.type": "snappy",
                    "message.max.bytes": 10_485_760,   ← 10 MB
                  }
        """
        raise NotImplementedError

    def publish(
        self,
        text: str,
        title: str,
        source: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> dict:
        """
        TODO 2: Generate doc_id = str(uuid.uuid4()) if not provided.

        TODO 3: Build the JSON payload dict:
                  {doc_id, text, title, source, metadata: metadata or {}, ts: time.time()}

        TODO 4: Call self._producer.produce() with:
                  topic=cfg.kafka_raw_topic,
                  key=doc_id.encode(),
                  value=json.dumps(payload).encode(),
                  on_delivery=self._on_delivery   ← raises on failure

        TODO 5: Call self._producer.flush() to block until delivery confirmed.
                Return {"doc_id": doc_id, "topic": ..., "partition": ..., "offset": ...}.

        Hint: Use a closure or instance variable to capture the delivery report result
              from the on_delivery callback.
        """
        raise NotImplementedError

    def _on_delivery(self, err, msg) -> None:
        """
        TODO 6: If err is not None, raise RuntimeError(f"Kafka delivery failed: {err}").
        """
        raise NotImplementedError

    def close(self) -> None:
        """TODO 7: Call self._producer.flush()."""
        raise NotImplementedError
