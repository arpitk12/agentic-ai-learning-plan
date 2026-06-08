"""
src/api/app.py — FastAPI application factory.

TODOs:
  1. implement create_app() — build FastAPI app with lifespan for startup/shutdown
  2. The lifespan should initialize all services (Neo4j, ChromaDB, Mem0, FallbackChain)
     and store them on app.state
"""
from __future__ import annotations
from contextlib import asynccontextmanager


# ── TODO 1: Application factory ───────────────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Startup: init all services. Shutdown: close connections."""
#     from src.config import get_config
#     from src.retrieval.vector_store import setup_store
#     from src.graph.neo4j_store import connect, get_schema
#     from src.memory.mem0_store import create_client
#     from src.resilience.fallback_chain import FallbackChain
#     from src.observability.cost_tracker import CostTracker
#     from src.observability.logger import get_logger
#     from src.agents.multimodal_agent import AgentDependencies
#
#     cfg = get_config()
#     logger = get_logger()
#
#     logger.info("starting up enterprise multimodal agent")
#
#     # Initialize services
#     collections = setup_store(cfg.chroma_persist_dir)
#     driver = connect(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
#     schema = get_schema(driver)
#     mem0_client = create_client(cfg.mem0_api_key)
#     chain = FallbackChain(cfg.llm_model_chain)
#     tracker = CostTracker()
#
#     app.state.deps = AgentDependencies(
#         collections=collections, neo4j_driver=driver, neo4j_schema=schema,
#         fallback_chain=chain, mem0_client=mem0_client,
#         cost_tracker=tracker, logger=logger,
#     )
#
#     yield   # ← app runs here
#
#     # Shutdown
#     driver.close()
#     logger.info("shutdown complete")
#
#
# def create_app() -> FastAPI:
#     from fastapi import FastAPI
#     from src.api.routes import router
#     from src.api.middleware import RateLimitMiddleware
#     from src.config import get_config
#
#     cfg = get_config()
#     app = FastAPI(
#         title="Enterprise Multimodal Compliance Agent",
#         description="Graph RAG + Guardrails + Long-term Memory",
#         version="1.0",
#         lifespan=lifespan,
#     )
#     app.add_middleware(RateLimitMiddleware, rpm=cfg.api_rate_limit_rpm)
#     app.include_router(router)
#     return app
#
#
# app = create_app()

raise NotImplementedError("Implement create_app() in src/api/app.py")
