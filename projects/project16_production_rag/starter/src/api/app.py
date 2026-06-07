"""
FastAPI application factory.

The lifespan handler is the critical design point: it loads the VectorStore
and HybridRetriever exactly once at startup and stores them on app.state so
every request handler can access them without globals.
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
    TODO 1: Create a VectorStore and a HybridRetriever (which builds the BM25 index)
    TODO 2: Log how many chunks are currently indexed
    TODO 3: Store both on app.state so routes can access them
    TODO 4: yield to hand control to the running server
    TODO 5: Log a shutdown message after the yield
    """
    raise NotImplementedError
    yield


def create_app() -> FastAPI:
    """
    TODO 6: Create a FastAPI instance using the lifespan handler and basic metadata
    TODO 7: Register RequestMiddleware
    TODO 8: Register the router
    TODO 9: Return the app
    """
    raise NotImplementedError


# Consumed by:  uvicorn src.api.app:app --reload
app = create_app()
