"""
TODO — Implement the FastAPI app factory with a lifespan.

Critical design: the HybridRetriever (and its BM25 index) is loaded ONCE
at startup and stored on app.state. Routes read it from there. This avoids
re-loading the ~384-dim embedding model on every request.

Pattern:
  @asynccontextmanager
  async def lifespan(app):
      store     = VectorStore()             # connect to ChromaDB
      retriever = HybridRetriever(store)    # builds BM25 at startup
      app.state.store     = store
      app.state.retriever = retriever
      yield                                 # app serves requests here
      # cleanup (nothing needed for ChromaDB)

The app factory (create_app) should:
  1. Create FastAPI(lifespan=lifespan, title=..., version=...)
  2. Add RequestMiddleware (from src.api.middleware)
  3. Include the router (from src.api.routes)
  4. Return the app

The module-level `app = create_app()` is what uvicorn imports.
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.config import cfg
from src.store.chroma_store import VectorStore
from src.retrieval.retriever import HybridRetriever
from src.api.middleware import RequestMiddleware
from src.api.routes import router

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    TODO 1: Create VectorStore() → store
    TODO 2: Create HybridRetriever(store) → retriever  (builds BM25 index)
    TODO 3: Log how many chunks are indexed
    TODO 4: Set app.state.store = store  and  app.state.retriever = retriever
    TODO 5: yield  (server is now live)
    TODO 6: Log shutdown message
    """
    raise NotImplementedError
    yield   # keep this line — remove the raise above and implement above the yield


def create_app() -> FastAPI:
    """
    TODO 7: Instantiate FastAPI(lifespan=lifespan, title="Production RAG Agent API",
                                version="1.0.0")
    TODO 8: app.add_middleware(RequestMiddleware)
    TODO 9: app.include_router(router)
    TODO 10: return app
    """
    raise NotImplementedError


# Consumed by:  uvicorn src.api.app:app --reload
app = create_app()
