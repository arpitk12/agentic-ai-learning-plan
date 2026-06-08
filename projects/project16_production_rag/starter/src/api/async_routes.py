"""
Async ingestion routes — non-blocking document ingestion via Celery.

Instead of embedding inside the API request, these routes push a job to
the Redis queue and return a job_id immediately (<5ms). The caller polls
GET /ingest/job/{job_id} until the worker finishes.

After the worker stores chunks it sets "bm25:stale" in Redis. The /query
route checks this flag and rebuilds the in-memory BM25 index transparently
before answering — no API restart needed.

Endpoints:
  POST /ingest/async               — submit text, get job_id back immediately
  POST /ingest/directory/async     — submit a directory path to the worker
  GET  /ingest/job/{job_id}        — poll status: pending/started/success/failure
  GET  /ingest/queue/stats         — live worker and queue statistics
"""
from __future__ import annotations
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from celery.result import AsyncResult

from src.ingestion.tasks import celery_app, ingest_text_task, ingest_directory_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["async-ingestion"])


class AsyncIngestTextRequest(BaseModel):
    text:   str
    title:  str = "Untitled"
    source: str = "api"


class AsyncIngestDirectoryRequest(BaseModel):
    source:           str
    replace_existing: bool = False


class JobStatusResponse(BaseModel):
    job_id:  str
    status:  str
    result:  dict | None = None
    error:   str | None  = None
    ready:   bool        = False


def _job_response(job_id: str) -> JobStatusResponse:
    """
    TODO 1: Create an AsyncResult from the job_id using celery_app
    TODO 2: Map the Celery state (SUCCESS/FAILURE/PENDING/STARTED) to a lowercase string
    TODO 3: If SUCCESS, set result = ar.result (the dict returned by the task)
    TODO 4: If FAILURE, set error = str(ar.result)
    TODO 5: Return a JobStatusResponse with job_id, status, result, error, ready
    """
    raise NotImplementedError


@router.post("/async", summary="Submit text for background ingestion")
async def ingest_text_async(body: AsyncIngestTextRequest, request: Request):
    """
    TODO 6: Call ingest_text_task.apply_async(kwargs={...}, queue="ingestion")
    TODO 7: Return {"job_id": job.id, "status": "pending"}
    TODO 8: On any exception raise HTTPException 503 with a helpful message
    """
    raise NotImplementedError


@router.post("/directory/async", summary="Submit a directory for background ingestion")
async def ingest_directory_async(body: AsyncIngestDirectoryRequest, request: Request):
    """
    TODO 9:  Call ingest_directory_task.apply_async(kwargs={...}, queue="ingestion")
    TODO 10: Return {"job_id": job.id, "status": "pending", "source": body.source}
    TODO 11: On any exception raise HTTPException 503
    """
    raise NotImplementedError


@router.get("/job/{job_id}", response_model=JobStatusResponse, summary="Poll job status")
async def get_job_status(job_id: str):
    """
    TODO 12: Call _job_response(job_id) and return the result
    TODO 13: Wrap in try/except and raise HTTPException 500 on error
    """
    raise NotImplementedError


@router.get("/queue/stats", summary="Worker and queue statistics")
async def queue_stats():
    """
    TODO 14: Use celery_app.control.inspect() to get active, reserved, and stats
    TODO 15: Return a dict with those three keys
    TODO 16: Raise HTTPException 503 if the workers are unreachable
    """
    raise NotImplementedError
