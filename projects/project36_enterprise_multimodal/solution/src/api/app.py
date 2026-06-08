"""
solution/src/api/app.py — Full implementation.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
import sys
import os

# Allow imports from solution/src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@asynccontextmanager
async def lifespan(app):
    from src.config import get_config  # type: ignore
    from src.retrieval.vector_store import setup_store  # type: ignore
    from src.resilience.fallback_chain import FallbackChain  # type: ignore
    from src.observability.cost_tracker import CostTracker  # type: ignore
    from src.agents.multimodal_agent import AgentDependencies  # type: ignore

    cfg = get_config()

    # ChromaDB (always available)
    collections = setup_store(cfg.chroma_persist_dir)

    # Neo4j (optional — graceful fallback if not running)
    driver, schema = None, "Node labels: Document, Entity\nRelationship types: MENTIONS, RELATION"
    try:
        from src.graph.neo4j_store import connect, get_schema  # type: ignore
        driver = connect(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
        schema = get_schema(driver)
    except Exception as e:
        print(f"[startup] Neo4j unavailable ({e}) — graph RAG disabled")

    # Mem0 memory
    mem0_client = None
    try:
        from src.memory.mem0_store import create_client  # type: ignore
        mem0_client = create_client(cfg.mem0_api_key)
    except Exception as e:
        print(f"[startup] Mem0 unavailable ({e}) — memory disabled")

    chain = FallbackChain(cfg.llm_model_chain)
    tracker = CostTracker()

    app.state.deps = AgentDependencies(
        collections=collections, neo4j_driver=driver, neo4j_schema=schema,
        fallback_chain=chain, mem0_client=mem0_client,
        cost_tracker=tracker, logger=None,
    )
    app.state.cfg = cfg

    print("[startup] Enterprise Multimodal Agent ready")
    yield

    if driver:
        driver.close()
    print("[shutdown] complete")


def create_app():
    from fastapi import FastAPI  # type: ignore
    from src.api.routes import router  # type: ignore
    from src.api.middleware import RateLimitMiddleware  # type: ignore
    from src.config import get_config  # type: ignore

    cfg = get_config()
    app = FastAPI(
        title="Enterprise Multimodal Compliance Agent",
        description="Graph RAG + 4-Layer Guardrails + Mem0 Long-term Memory",
        version="1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RateLimitMiddleware, rpm=cfg.api_rate_limit_rpm)
    app.include_router(router)
    return app


app = create_app()
