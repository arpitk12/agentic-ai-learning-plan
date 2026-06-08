"""
FastAPI application — lifespan loads all heavy objects once at startup.

Startup sequence
────────────────
  1. Connect Qdrant store
  2. Load embedding model (sentence-transformers, ~90 MB)
  3. Load reranker cross-encoder (MS-MARCO MiniLM, ~23 MB)
  4. Load NLI faithfulness checker (DeBERTa, ~750 MB) — longest step
  5. Connect Redis cache
  6. Build semantic cache (in-memory)
  7. Register routers

All heavy objects live on app.state and are injected into routes via request.app.state.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from sentence_transformers import CrossEncoder

from src.agents.rag_agent import RAGAgent
from src.cache.redis_cache import RedisCache
from src.cache.semantic_cache import SemanticCache
from src.config import cfg
from src.hallucination.abstain_policy import AbstainPolicy
from src.hallucination.citation_verifier import CitationVerifier
from src.hallucination.faithfulness_checker import FaithfulnessChecker
from src.ingestion.embedder import Embedder
from src.retrieval.retriever import HybridRetriever
from src.store.qdrant_store import QdrantStore
from src.api.routes import router
from src.api.middleware import RequestIDMiddleware, TimingMiddleware, RateLimitMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise RAG API...")

    # Vector store
    store = QdrantStore()
    store.create_collection()

    # Embedding model
    embedder = Embedder()

    # Cross-encoder reranker
    reranker = CrossEncoder(cfg.reranker_model)

    # Faithfulness checker (loads NLI model — ~750 MB download on first run)
    checker = FaithfulnessChecker()

    # Cache
    redis_cache = RedisCache()
    sem_cache = SemanticCache()

    # Hybrid retriever
    retriever = HybridRetriever(qdrant_store=store, embedder=embedder)

    # RAG agent
    agent = RAGAgent(
        retriever=retriever,
        embedder=embedder,
        faithfulness_checker=checker,
        reranker_model=reranker,
        redis_cache=redis_cache,
        semantic_cache=sem_cache,
    )

    # Attach to app state
    app.state.store = store
    app.state.embedder = embedder
    app.state.retriever = retriever
    app.state.agent = agent
    app.state.redis_cache = redis_cache
    app.state.semantic_cache = sem_cache

    logger.info("Enterprise RAG API ready")
    yield

    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG API",
        description="10M-doc RAG with zero-hallucination guarantee",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware (outermost first)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Routes
    app.include_router(router)

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_app()
