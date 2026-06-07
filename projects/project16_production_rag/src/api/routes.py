"""
API routes:
  POST /query   — answer a question via the orchestrator (RAG or direct)
  POST /ingest  — ingest a text snippet at runtime; rebuilds BM25 index
  GET  /health  — liveness + chunk count
  GET  /stats   — vector-store stats
  GET  /eval    — run evaluation suite; returns EvalReport
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


# ── helpers ─────────────────────────────────────────────────────────────────

def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "?")


def _get_retriever(request: Request):
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")
    return retriever


def _get_store(request: Request):
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised")
    return store


# ── routes ───────────────────────────────────────────────────────────────────

@router.get("/health", tags=["ops"])
async def health(request: Request):
    """Liveness probe — always fast."""
    try:
        store = _get_store(request)
        n = store.count()
        return {"status": "ok", "chunks_indexed": n}
    except Exception:
        return {"status": "ok", "chunks_indexed": -1}


@router.get("/stats", tags=["ops"])
async def stats(request: Request):
    """Return vector-store statistics."""
    store = _get_store(request)
    return store.stats()


@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(body: QueryRequest, request: Request):
    """
    Answer a question.  The orchestrator decides whether to use
    retrieval-augmented generation or a direct LLM call.
    """
    rid = _request_id(request)
    retriever = _get_retriever(request)
    logger.info("[%s] /query question=%r", rid, body.question[:80])
    try:
        return await handle(body, retriever, rid)
    except Exception as exc:
        logger.exception("[%s] /query error: %s", rid, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest", response_model=IngestionResult, tags=["ingestion"])
async def ingest(
    request: Request,
    text: str,
    title: str = "Untitled",
    source: str = "api",
):
    """
    Ingest a text snippet at runtime.
    Stores chunks in ChromaDB and rebuilds the BM25 index.
    """
    rid = _request_id(request)
    store     = _get_store(request)
    retriever = _get_retriever(request)

    logger.info("[%s] /ingest title=%r len=%d", rid, title, len(text))
    try:
        result = ingest_text(text=text, title=title, source=source, store=store)
        retriever.rebuild_bm25()
        logger.info("[%s] ingested %d chunks (total %d)", rid, result.chunks_created, store.count())
        return result
    except Exception as exc:
        logger.exception("[%s] /ingest error: %s", rid, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/eval", tags=["eval"])
async def eval_endpoint(request: Request):
    """
    Run the built-in golden evaluation suite.
    Returns an EvalReport (faithfulness + relevancy per case + aggregate).
    """
    rid       = _request_id(request)
    retriever = _get_retriever(request)
    logger.info("[%s] /eval started", rid)
    try:
        report = await run_eval(retriever, request_id=rid)
        return report.model_dump()
    except Exception as exc:
        logger.exception("[%s] /eval error: %s", rid, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
