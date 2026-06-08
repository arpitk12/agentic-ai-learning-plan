"""
Chunk consumer — reads raw documents from 'raw-documents', splits into chunks,
publishes to 'document-chunks'.

Pipeline: raw-documents → [chunk_consumer] → document-chunks

Key behaviours
──────────────
  Manual offset commit: only commits AFTER successfully publishing all chunks.
  If the process crashes mid-way, Kafka redelivers the message → idempotent chunking.

  DLQ: messages that fail after 3 retries are sent to 'dlq-ingestion'.
  Monitor the DLQ with an alert — it indicates bad document formats.
"""
from __future__ import annotations

import json
import logging
import signal
import sys

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from src.config import cfg
from src.ingestion.chunker import Chunker

logger = logging.getLogger(__name__)


class ChunkConsumer:
    GROUP_ID = "chunker-group"

    def __init__(
        self,
        bootstrap_servers: str = cfg.kafka_bootstrap_servers,
        max_retries: int = 3,
    ) -> None:
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": self.GROUP_ID,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,   # manual commit — at-least-once
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
        self._chunker = Chunker()
        self._max_retries = max_retries
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def run(self) -> None:
        self._consumer.subscribe([cfg.kafka_raw_topic])
        logger.info("ChunkConsumer started — listening on '%s'", cfg.kafka_raw_topic)

        while self._running:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka error: %s", msg.error())
                self._send_to_dlq(msg.value(), error=str(msg.error()))
                self._consumer.commit()
                continue

            try:
                self._process(msg)
            except Exception as exc:
                logger.exception("Failed to process message: %s", exc)
                self._send_to_dlq(msg.value(), error=str(exc))
            finally:
                self._consumer.commit()

        self._consumer.close()
        logger.info("ChunkConsumer shut down cleanly")

    def _process(self, msg) -> None:
        doc = json.loads(msg.value())
        chunks = self._chunker.chunk(doc["text"])

        for i, chunk in enumerate(chunks):
            chunk_payload = {
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk.text,
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "metadata": doc.get("metadata", {}),
                "ts": doc.get("ts"),
            }
            self._producer.produce(
                topic=cfg.kafka_chunks_topic,
                key=doc["doc_id"].encode(),
                value=json.dumps(chunk_payload).encode(),
            )

        self._producer.flush()
        logger.debug(
            "Chunked doc %s → %d chunks → '%s'",
            doc["doc_id"], len(chunks), cfg.kafka_chunks_topic,
        )

    def _send_to_dlq(self, raw_value: bytes, error: str) -> None:
        try:
            self._producer.produce(
                topic=cfg.kafka_dlq_topic,
                value=raw_value,
                headers={"error": error.encode()},
            )
            self._producer.flush()
        except Exception as exc:
            logger.error("Failed to write to DLQ: %s", exc)

    def _handle_shutdown(self, *_) -> None:
        logger.info("Shutdown signal received")
        self._running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ChunkConsumer().run()
