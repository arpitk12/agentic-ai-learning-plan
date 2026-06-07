"""
TODO — Implement all API routes.

Endpoints to build:
  GET  /health  → {"status": "ok", "chunks_indexed": N}
  GET  /stats   → store.stats() dict
  POST /query   → orchestrator.handle(request, retriever)  → QueryResponse
  POST /ingest  → ingest_text(...) then retriever.rebuild_bm25() → IngestionResult
  GET  /eval    → evaluator.run_eval(retriever) → EvalReport dict

Tip: access shared state via request.app.state.store / request.app.state.retriever
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
    TODO 1: Get store from request.app.state.store
    TODO 2: Return {"status": "ok", "chunks_indexed": store.count()}
    TODO 3: Wrap in try/except — on error return {"status": "ok", "chunks_indexed": -1}
    """
    raise NotImplementedError


@router.get("/stats", tags=["ops"])
async def stats(request: Request):
    """
    TODO 4: Get store from request.app.state.store
    TODO 5: Return store.stats()
    """
    raise NotImplementedError


@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(body: QueryRequest, request: Request):
    """
    TODO 6: Get retriever from request.app.state.retriever
    TODO 7: Return await handle(body, retriever, _rid(request))
    TODO 8: Wrap in try/except → raise HTTPException(status_code=500, detail=str(exc))
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
    TODO 9:  Get store + retriever from request.app.state
    TODO 10: result = ingest_text(text=text, title=title, source=source, store=store)
    TODO 11: retriever.rebuild_bm25()   ← important! refreshes the keyword index
    TODO 12: return result
    TODO 13: Wrap in try/except → raise HTTPException(500, ...)
    """
    raise NotImplementedError


@router.get("/eval", tags=["eval"])
async def eval_endpoint(request: Request):
    """
    TODO 14: Get retriever from request.app.state.retriever
    TODO 15: report = await run_eval(retriever, request_id=_rid(request))
    TODO 16: return report.model_dump()
    """
    raise NotImplementedError
