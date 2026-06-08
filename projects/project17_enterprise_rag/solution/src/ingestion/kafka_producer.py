"""
Kafka producer — publishes raw documents to the 'raw-documents' topic.

Design choices
──────────────
  acks="all"      — wait for all in-sync replicas before ack (no data loss)
  retries=5       — auto-retry transient failures
  compression     — snappy compression reduces network bandwidth by ~50%
  key=doc_id      — same doc always goes to same partition (ordering guarantee)
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from confluent_kafka import Producer

from src.config import cfg

logger = logging.getLogger(__name__)


class DocumentProducer:
    def __init__(self, bootstrap_servers: str = cfg.kafka_bootstrap_servers) -> None:
        self._producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "acks": "all",
                "retries": 5,
                "retry.backoff.ms": 500,
                "compression.type": "snappy",
                "message.max.bytes": 10_485_760,  # 10 MB
            }
        )

    def publish(
        self,
        text: str,
        title: str,
        source: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> dict:
        """
        Publish a raw document to the raw-documents topic.
        Returns a dict with {doc_id, topic, partition, offset}.
        """
        doc_id = doc_id or str(uuid.uuid4())
        payload = json.dumps(
            {
                "doc_id": doc_id,
                "text": text,
                "title": title,
                "source": source,
                "metadata": metadata or {},
                "ts": time.time(),
            }
        ).encode()

        result: dict = {}

        def _on_delivery(err, msg):
            if err:
                raise RuntimeError(f"Kafka delivery failed: {err}")
            result.update(
                topic=msg.topic(),
                partition=msg.partition(),
                offset=msg.offset(),
            )

        self._producer.produce(
            topic=cfg.kafka_raw_topic,
            key=doc_id.encode(),
            value=payload,
            on_delivery=_on_delivery,
        )
        self._producer.flush()

        logger.info("Published doc %s → %s p%d o%d", doc_id, result.get("topic"), result.get("partition"), result.get("offset"))
        return {"doc_id": doc_id, **result}

    def close(self) -> None:
        self._producer.flush()
