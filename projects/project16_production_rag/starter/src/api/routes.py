"""
API route handlers.

All handlers read shared state from request.app.state.store and
request.app.state.retriever — objects loaded once at startup by the lifespan.

Endpoints:
  GET  /health  — liveness probe, always fast
  GET  /stats   — vector store statistics
  POST /query   — answer a question via the orchestrator
  POST /ingest  — ingest a text snippet and rebuild the BM25 index
  GET  /eval    — run the evaluation suite
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Request, HTTPException
from src.models import QueryRequest, QueryResponse, IngestionResult
from src.agents.orchestrator import handle
from src.ingestion.pipeline import ingest_text
from src.evaluation.evaluator import run_eval

logger = logging.getLogger(__name__)
router = APIRouter()


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "?")


@router.get("/health", tags=["ops"])
async def health(request: Request):
    """
    TODO 1: Return the current status and indexed chunk count.
            Return a safe fallback if the store is unavailable.
    """
    raise NotImplementedError


@router.get("/stats", tags=["ops"])
async def stats(request: Request):
    """TODO 2: Return vector store statistics."""
    raise NotImplementedError


@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(body: QueryRequest, request: Request):
    """
    TODO 3: Pass the request to the orchestrator and return the response.
            Raise a 500 error on any failure.
    """
    raise NotImplementedError


@router.post("/ingest", response_model=IngestionResult, tags=["ingestion"])
async def ingest(
    request: Request,
    text: str,
    title: str = "Untitled",
    source: str = "api",
):
    """
    TODO 4: Ingest the text into the vector store.
    TODO 5: Rebuild the BM25 index so the new content is immediately searchable.
            Raise a 500 error on any failure.
    """
    raise NotImplementedError


@router.get("/eval", tags=["eval"])
async def eval_endpoint(request: Request):
    """TODO 6: Run the evaluation suite and return the report as a dict."""
    raise NotImplementedError
