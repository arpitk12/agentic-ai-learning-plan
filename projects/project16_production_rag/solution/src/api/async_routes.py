"""
Async ingestion routes — fire-and-forget document ingestion via Celery.

Why async?
  The synchronous POST /ingest blocks the API thread for the entire
  embed-and-store pipeline (~200ms–2s per document). Under load that
  exhausts the worker pool quickly.

  These routes return immediately with a job_id. The caller polls
  GET /ingest/job/{job_id} until status becomes "success" or "failure".

Endpoints:
  POST /ingest/async            — submit a text snippet for background ingestion
  POST /ingest/directory/async  — submit a directory path (worker must have access)
  GET  /ingest/job/{job_id}     — poll job status + result

BM25 freshness:
  After the worker stores new chunks it sets "bm25:stale" in Redis.
  The /query route checks this flag and rebuilds the in-memory BM25 index
  before answering — so new content is always searchable on the next query
  after ingestion completes, without the API knowing about the worker directly.
"""
from __future__ import annotations
import logging
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from celery.result import AsyncResult

from src.ingestion.tasks import celery_app, ingest_text_task, ingest_directory_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["async-ingestion"])

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ── request / response models ─────────────────────────────────────────────────

class AsyncIngestTextRequest(BaseModel):
    text:   str
    title:  str = "Untitled"
    source: str = "api"


class AsyncIngestDirectoryRequest(BaseModel):
    source:           str           # absolute or relative path the *worker* can read
    replace_existing: bool = False


class JobStatusResponse(BaseModel):
    job_id:  str
    status:  str                    # pending | started | success | failure | retry
    result:  dict | None = None     # IngestionResult dict when status == success
    error:   str | None  = None     # error message when status == failure
    ready:   bool        = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _celery_available() -> bool:
    """
    Cheap check: try to ping the broker.
    Returns False if Redis is not reachable (graceful degradation).
    """
    try:
        celery_app.control.inspect(timeout=0.5).ping()
        return True
    except Exception:
        return False


def _job_response(job_id: str) -> JobStatusResponse:
    """Convert a Celery AsyncResult into a clean JobStatusResponse."""
    ar     = AsyncResult(job_id, app=celery_app)
    status = ar.state.lower()         # celery states are uppercase

    result = None
    error  = None

    if ar.state == "SUCCESS":
        result = ar.result            # dict from task return value
    elif ar.state == "FAILURE":
        error  = str(ar.result)       # exception repr

    return JobStatusResponse(
        job_id = job_id,
        status = status,
        result = result,
        error  = error,
        ready  = ar.ready(),
    )


# ── routes ────────────────────────────────────────────────────────────────────

@router.post("/async", summary="Submit text for background ingestion")
async def ingest_text_async(body: AsyncIngestTextRequest, request: Request):
    """
    Submit a text snippet to the Celery ingestion queue.
    Returns immediately with a job_id — ingestion runs in the background.

    Poll GET /ingest/job/{job_id} to track progress.
    """
    logger.info(
        "Submitting async ingest: title=%r len=%d", body.title, len(body.text)
    )
    try:
        job = ingest_text_task.apply_async(
            kwargs={"text": body.text, "title": body.title, "source": body.source},
            queue="ingestion",
        )
        return {"job_id": job.id, "status": "pending", "queue": "ingestion"}
    except Exception as exc:
        logger.exception("Failed to submit ingestion task: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not connect to message queue. "
                "Is Redis running? Use POST /ingest for synchronous ingestion."
            ),
        ) from exc


@router.post("/directory/async", summary="Submit a directory for background ingestion")
async def ingest_directory_async(body: AsyncIngestDirectoryRequest, request: Request):
    """
    Submit a directory path to the Celery ingestion queue.

    The worker process (not the API) must have read access to the path.
    In Docker Compose the path must be a shared volume mount.

    Example body:
      {"source": "/app/data/new_docs", "replace_existing": true}
    """
    logger.info("Submitting async directory ingest: source=%r", body.source)
    try:
        job = ingest_directory_task.apply_async(
            kwargs={
                "source":           body.source,
                "replace_existing": body.replace_existing,
            },
            queue="ingestion",
        )
        return {"job_id": job.id, "status": "pending", "source": body.source}
    except Exception as exc:
        logger.exception("Failed to submit directory ingestion task: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/job/{job_id}", response_model=JobStatusResponse, summary="Poll job status")
async def get_job_status(job_id: str):
    """
    Check the status of an async ingestion job.

    Possible status values:
      pending  — job queued but not yet picked up by a worker
      started  — worker is actively running the pipeline
      success  — pipeline complete; result contains IngestionResult
      failure  — pipeline failed; error contains the exception message
      retry    — transient error, worker will retry automatically

    Tip: poll every 1–2 seconds until ready == true.
    """
    try:
        return _job_response(job_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/queue/stats", summary="Queue and worker statistics")
async def queue_stats():
    """
    Return live statistics from the Celery/Redis queue.
    Useful for monitoring: active tasks, reserved tasks, worker count.
    """
    try:
        i = celery_app.control.inspect(timeout=1.0)
        return {
            "active":   i.active()   or {},   # tasks currently executing
            "reserved": i.reserved() or {},   # tasks queued to a worker but not started
            "stats":    i.stats()    or {},    # worker statistics (concurrency, etc.)
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach workers: {exc}",
        ) from exc
