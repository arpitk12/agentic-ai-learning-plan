"""
Celery task definitions — async ingestion pipeline.

Architecture:
  API process  →  Redis queue  →  Celery worker process  →  ChromaDB
                  (broker)

The worker is a completely separate process (or container). It:
  1. Picks up an ingestion job from the Redis queue
  2. Runs the full pipeline: text → chunk → embed → store
  3. Sets a Redis flag ("bm25:stale") so the API knows to rebuild its
     in-memory BM25 index before the next query

Usage:
  # Start the worker (separate terminal / Docker container)
  celery -A src.ingestion.tasks worker --loglevel=info --concurrency=2

  # Submit a job programmatically
  from src.ingestion.tasks import ingest_text_task
  job = ingest_text_task.delay(text="...", title="Doc", source="api")
  print(job.id)   # job ID for polling

Environment variables:
  REDIS_URL   — broker + result backend  (default: redis://localhost:6379/0)
  CHROMA_HOST — passed through to VectorStore (same as API)
"""
from __future__ import annotations
import logging
import os

from celery import Celery

logger = logging.getLogger(__name__)

# ── Celery app ────────────────────────────────────────────────────────────────

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rag_ingestion",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
celery_app.conf.update(
    # Serialisation
    task_serializer   = "json",
    result_serializer = "json",
    accept_content    = ["json"],

    # Reliability
    task_acks_late    = True,         # ack only after task completes (not on receipt)
    task_track_started= True,         # STARTED state is reported to backend
    worker_prefetch_multiplier = 1,   # one task at a time per worker thread

    # Result TTL
    result_expires    = 3600,         # results kept in Redis for 1 hour

    # Retry defaults
    task_max_retries  = 3,
    task_default_retry_delay = 30,    # 30s between retries

    # Routing — all ingestion tasks go to a dedicated queue
    task_default_queue = "ingestion",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mark_bm25_stale() -> None:
    """
    Set a Redis flag so the API knows to rebuild its in-memory BM25 index.

    The API checks this flag in the /query route (via _check_bm25_staleness).
    Setting ex=7200 means the flag expires after 2h even if the API is down.
    Non-fatal: if Redis is unreachable, the flag is silently skipped.
    """
    try:
        import redis as _redis
        r = _redis.from_url(REDIS_URL, decode_responses=True)
        r.set("bm25:stale", "1", ex=7200)
        logger.debug("bm25:stale flag set in Redis")
    except Exception as exc:
        logger.warning("Could not set bm25:stale flag: %s", exc)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@celery_app.task(
    bind        = True,
    name        = "ingestion.ingest_text",
    max_retries = 3,
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
    Background task: ingest a single text snippet.

    Flow: text → chunk → embed → store in ChromaDB → mark BM25 stale
    Retries up to 3 times on transient errors (e.g. ChromaDB unavailable).

    Returns a dict that matches IngestionResult.model_dump().
    """
    logger.info("[task:%s] ingesting title=%r len=%d", self.request.id, title, len(text))
    try:
        # Import here (not at module level) so the worker process doesn't
        # need to load the embedding model until a task actually arrives.
        from src.store.chroma_store import VectorStore
        from src.ingestion.pipeline import ingest_text

        store  = VectorStore()
        result = ingest_text(text=text, title=title, source=source, store=store)
        _mark_bm25_stale()

        logger.info(
            "[task:%s] done — %d chunks stored",
            self.request.id, result.chunks_stored,
        )
        return result.model_dump()

    except Exception as exc:
        logger.warning("[task:%s] error: %s — retrying", self.request.id, exc)
        raise self.retry(exc=exc)


@celery_app.task(
    bind        = True,
    name        = "ingestion.ingest_directory",
    max_retries = 2,
    default_retry_delay = 60,
)
def ingest_directory_task(
    self,
    *,
    source:           str,
    replace_existing: bool = False,
) -> dict:
    """
    Background task: ingest all .md/.txt files in a directory.

    Intended for use with file-system watchers or webhook triggers
    (e.g. trigger this when a new document is pushed to S3 / GCS / a repo).

    Returns a dict matching IngestionResult.model_dump().
    """
    logger.info(
        "[task:%s] ingesting directory=%r replace=%s",
        self.request.id, source, replace_existing,
    )
    try:
        from src.store.chroma_store import VectorStore
        from src.ingestion.pipeline import ingest_directory

        store  = VectorStore()
        result = ingest_directory(
            source=source,
            store=store,
            replace_existing=replace_existing,
        )
        _mark_bm25_stale()

        logger.info(
            "[task:%s] done — %d docs, %d chunks",
            self.request.id, result.docs_loaded, result.chunks_stored,
        )
        return result.model_dump()

    except Exception as exc:
        logger.warning("[task:%s] error: %s — retrying", self.request.id, exc)
        raise self.retry(exc=exc)
