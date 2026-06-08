"""
Celery task definitions — async ingestion pipeline.

The worker is a completely separate process (or Docker container) that:
  1. Picks up an ingestion job from the Redis queue
  2. Runs the full pipeline: text → chunk → embed → store in ChromaDB
  3. Sets a Redis flag ("bm25:stale") so the API knows to rebuild its
     in-memory BM25 index before the next query

Why a separate worker?
  Embedding is CPU-heavy (~200ms–2s per document). Running it inside the
  API process blocks request handling. The queue decouples ingestion
  throughput from query latency.

Start the worker:
  celery -A src.ingestion.tasks worker --loglevel=info --queues=ingestion --concurrency=2

Environment:
  REDIS_URL — broker + result backend (default: redis://localhost:6379/0)
"""
from __future__ import annotations
import logging
import os

from celery import Celery

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rag_ingestion",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
celery_app.conf.update(
    task_serializer           = "json",
    result_serializer         = "json",
    accept_content            = ["json"],
    task_acks_late            = True,
    task_track_started        = True,
    worker_prefetch_multiplier= 1,
    result_expires            = 3600,
    task_default_queue        = "ingestion",
)


def _mark_bm25_stale() -> None:
    """
    TODO 1: Connect to Redis using REDIS_URL
    TODO 2: Set the key "bm25:stale" to "1" with a 2-hour expiry
    TODO 3: Catch any connection error silently — this flag is best-effort
    """
    raise NotImplementedError


@celery_app.task(
    bind             = True,
    name             = "ingestion.ingest_text",
    max_retries      = 3,
    default_retry_delay = 30,
)
def ingest_text_task(
    self,
    *,
    text:   str,
    title:  str = "Untitled",
    source: str = "worker",
) -> dict:
    """
    Background task: chunk, embed, and store a single text snippet.
    Returns a dict matching IngestionResult.model_dump().

    TODO 4: Import VectorStore and ingest_text inside the function body
            (lazy import — avoids loading the embedding model at worker startup)
    TODO 5: Create a VectorStore and call ingest_text
    TODO 6: Call _mark_bm25_stale() so the API rebuilds its index
    TODO 7: Return result.model_dump()
    TODO 8: On any exception call self.retry(exc=exc) to re-queue with backoff
    """
    raise NotImplementedError


@celery_app.task(
    bind             = True,
    name             = "ingestion.ingest_directory",
    max_retries      = 2,
    default_retry_delay = 60,
)
def ingest_directory_task(
    self,
    *,
    source:           str,
    replace_existing: bool = False,
) -> dict:
    """
    Background task: ingest all documents in a directory.
    Used with file-system watchers or webhook triggers.

    TODO 9:  Import VectorStore and ingest_directory lazily
    TODO 10: Run the directory ingestion pipeline
    TODO 11: Call _mark_bm25_stale()
    TODO 12: Return result.model_dump()
    TODO 13: On any exception call self.retry(exc=exc)
    """
    raise NotImplementedError
