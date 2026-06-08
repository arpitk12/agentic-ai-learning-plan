"""FastAPI application — lifespan initialises all heavy objects at startup."""
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
from src.hallucination.faithfulness_checker import FaithfulnessChecker
from src.ingestion.embedder import Embedder
from src.retrieval.retriever import HybridRetriever
from src.store.qdrant_store import QdrantStore
from src.api.routes import router
from src.api.middleware import RequestIDMiddleware, TimingMiddleware, RateLimitMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    TODO 1: Initialise QdrantStore, call create_collection().

    TODO 2: Initialise Embedder (loads sentence-transformers model).

    TODO 3: Initialise CrossEncoder(cfg.reranker_model) as reranker.

    TODO 4: Initialise FaithfulnessChecker() (loads NLI model — 750 MB on first run).

    TODO 5: Initialise RedisCache() and SemanticCache().

    TODO 6: Initialise HybridRetriever(qdrant_store=store, embedder=embedder).

    TODO 7: Initialise RAGAgent with all components.

    TODO 8: Attach everything to app.state:
              app.state.store, app.state.embedder, app.state.retriever,
              app.state.agent, app.state.redis_cache, app.state.semantic_cache

    After all assignments, yield (API is now serving).
    """
    raise NotImplementedError
    yield


def create_app() -> FastAPI:
    """
    TODO 9: Create FastAPI app with title, description, version, lifespan.
            Add middleware: RateLimitMiddleware, TimingMiddleware, RequestIDMiddleware.
            Include router.
            Mount Prometheus metrics at "/metrics" using make_asgi_app().
            Return app.
    """
    raise NotImplementedError


app = create_app()
