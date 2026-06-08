"""
FastAPI application factory.

Startup (lifespan):
  1. Create VectorStore (connects to ChromaDB — local or HTTP)
  2. Create HybridRetriever and build BM25 index from existing chunks
  3. Store both on app.state so routes can access them without globals
  4. Store the Redis URL on app.state so routes can check BM25 staleness

Shutdown:
  Nothing to clean up (ChromaDB handles its own file flushing).

BM25 staleness (async ingestion):
  When the Celery worker finishes ingesting a document it sets a Redis key
  "bm25:stale = 1".  The /query route checks this flag and rebuilds the
  in-memory BM25 index before answering — so newly ingested content is
  immediately searchable without a restart.
"""
from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.config import cfg
from src.store.chroma_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.api.middleware import RequestMiddleware
from src.api.routes import router
from src.api.async_routes import router as async_router

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ── lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy objects once at startup and share via app.state."""
    logger.info("Starting up — connecting to ChromaDB …")
    store     = VectorStore()
    retriever = HybridRetriever(store)
    n_chunks  = store.count()
    logger.info("ChromaDB ready — %d chunks indexed", n_chunks)

    app.state.store     = store
    app.state.retriever = retriever
    app.state.redis_url = REDIS_URL      # used by /query for BM25 staleness check

    yield  # server is running

    logger.info("Shutting down")


# ── app factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Production RAG Agent API",
        description=(
            "Modular RAG system with hybrid search (BM25 + vector), "
            "LLM reranking, multi-agent orchestration, and MCP tooling."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware (order matters: outermost first)
    app.add_middleware(RequestMiddleware)

    # Routes
    app.include_router(router)
    app.include_router(async_router)   # POST /ingest/async, GET /ingest/job/{id}

    return app


# Module-level instance consumed by uvicorn / Dockerfile CMD
app = create_app()
