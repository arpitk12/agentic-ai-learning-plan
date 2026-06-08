"""API route handlers."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request

from src.models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from src.observability.metrics import (
    ABSTAIN_COUNTER,
    CACHE_HITS,
    QUERY_COUNTER,
    QUERY_LATENCY,
    FAITHFULNESS_HISTOGRAM,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    store = request.app.state.store
    redis = request.app.state.redis_cache
    return HealthResponse(
        status="ok",
        qdrant="healthy" if store.is_healthy() else "unhealthy",
        redis="healthy" if redis.is_healthy() else "unhealthy",
        kafka="unknown",  # consumers run separately; API doesn't poll Kafka
    )


@router.get("/stats", response_model=StatsResponse, tags=["ops"])
async def stats(request: Request) -> StatsResponse:
    store = request.app.state.store
    redis = request.app.state.redis_cache
    sem = request.app.state.semantic_cache
    info = store.get_info()
    return StatsResponse(
        vectors_count=info["vectors_count"],
        query_cache_size=redis.query_cache_size(),
        semantic_cache_size=sem.size,
        embed_cache_hit_rate=sem.hit_rate,
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    agent = request.app.state.agent

    try:
        response = agent.answer(body, request_id=request_id)
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        QUERY_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc))

    # Prometheus metrics
    QUERY_COUNTER.labels(status="abstained" if response.abstained else "success").inc()
    QUERY_LATENCY.observe(response.latency_ms / 1000)
    if response.abstained:
        ABSTAIN_COUNTER.labels(reason=response.abstain_reason or "unknown").inc()
    if not response.abstained:
        FAITHFULNESS_HISTOGRAM.observe(response.faithfulness_score)
    if response.cached:
        CACHE_HITS.labels(tier="exact_or_semantic").inc()

    return response


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    """Publish a document to Kafka — returns immediately, indexing is async."""
    from src.ingestion.kafka_producer import DocumentProducer
    from src.config import cfg

    try:
        producer = DocumentProducer(cfg.kafka_bootstrap_servers)
        result = producer.publish(
            text=body.text,
            title=body.title,
            source=body.source,
            metadata=body.metadata,
        )
        return IngestResponse(
            doc_id=result["doc_id"],
            topic=result.get("topic", cfg.kafka_raw_topic),
            partition=result.get("partition", 0),
            offset=result.get("offset", -1),
        )
    except Exception as exc:
        logger.exception("Ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
