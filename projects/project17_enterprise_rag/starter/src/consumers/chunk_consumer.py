"""
Chunk consumer — reads raw-documents, splits into chunks, publishes to document-chunks.
See GUIDE.md Phase 2.4 Step 3 for the full design.
"""
from __future__ import annotations

import json
import logging
import signal

from src.config import cfg
from src.ingestion.chunker import Chunker

logger = logging.getLogger(__name__)


class ChunkConsumer:
    GROUP_ID = "chunker-group"

    def __init__(self, bootstrap_servers: str = cfg.kafka_bootstrap_servers) -> None:
        """
        TODO 1: Import confluent_kafka.Consumer and Producer.
                Create self._consumer with:
                  {"bootstrap.servers": ..., "group.id": self.GROUP_ID,
                   "auto.offset.reset": "earliest", "enable.auto.commit": False}
                Create self._producer with acks="all".
                Create self._chunker = Chunker().
                Set self._running = True.
                Register SIGTERM and SIGINT handlers to set self._running = False.
        """
        raise NotImplementedError

    def run(self) -> None:
        """
        TODO 2: Subscribe consumer to [cfg.kafka_raw_topic].

        TODO 3: Poll loop:
                  while self._running:
                    msg = self._consumer.poll(timeout=1.0)
                    if msg is None: continue
                    if msg.error(): send to DLQ, commit, continue
                    try: self._process(msg)
                    except: send to DLQ
                    finally: self._consumer.commit()

        TODO 4: Call self._consumer.close() after the loop.
        """
        raise NotImplementedError

    def _process(self, msg) -> None:
        """
        TODO 5: Parse doc = json.loads(msg.value()).
                Call self._chunker.chunk(doc["text"]) → list of Chunk objects.

        TODO 6: For each chunk, publish to cfg.kafka_chunks_topic with payload:
                  {doc_id, chunk_id: f"{doc_id}_{i}", chunk_index: i,
                   total_chunks, text: chunk.text, title, source, metadata, ts}

        TODO 7: Call self._producer.flush() after publishing all chunks.
        """
        raise NotImplementedError

    def _send_to_dlq(self, raw_value: bytes, error: str) -> None:
        """
        TODO 8: Publish raw_value to cfg.kafka_dlq_topic with error in headers.
                Catch any exception (DLQ failure must not crash the consumer).
        """
        raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ChunkConsumer().run()
