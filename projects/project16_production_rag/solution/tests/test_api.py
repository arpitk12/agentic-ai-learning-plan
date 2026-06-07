"""
Unit tests — FastAPI routes (uses TestClient, mocks LLM calls).

Run:
    pytest tests/test_api.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from src.models import QueryResponse, Citation, IngestionResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_store():
    """Minimal VectorStore mock."""
    store = MagicMock()
    store.count.return_value = 42
    store.stats.return_value = {"chunks": 42, "documents": 3}
    return store


@pytest.fixture(scope="module")
def mock_retriever():
    """Minimal HybridRetriever mock."""
    retriever = MagicMock()
    retriever.rebuild_bm25.return_value = None
    return retriever


@pytest.fixture(scope="module")
def client(mock_store, mock_retriever):
    """
    Build the FastAPI app, inject mocked state, return TestClient.
    We patch the lifespan so it doesn't try to connect to ChromaDB.
    """
    from src.api.app import create_app
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _mock_lifespan(app):
        app.state.store     = mock_store
        app.state.retriever = mock_retriever
        yield

    with patch("src.api.app.lifespan", _mock_lifespan):
        app = create_app()

    # Override state after creation (lifespan won't run in TestClient by default
    # unless you use app.router.lifespan_context)
    app.state.store     = mock_store
    app.state.retriever = mock_retriever

    return TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_contains_status_ok(self, client):
        r = client.get("/health")
        body = r.json()
        assert body.get("status") == "ok"

    def test_health_contains_chunks_indexed(self, client):
        r = client.get("/health")
        body = r.json()
        assert "chunks_indexed" in body


# ─────────────────────────────────────────────────────────────────────────────
# /stats
# ─────────────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_returns_200(self, client):
        r = client.get("/stats")
        assert r.status_code == 200

    def test_stats_contains_chunks(self, client):
        r = client.get("/stats")
        body = r.json()
        assert "chunks" in body


# ─────────────────────────────────────────────────────────────────────────────
# /query
# ─────────────────────────────────────────────────────────────────────────────

class TestQuery:
    def _mock_response(self) -> QueryResponse:
        return QueryResponse(
            answer="TechFlow has three tiers: Starter, Professional, Enterprise.",
            model="gpt-4o-mini",
            citations=[
                Citation(chunk_id="c1", text="TechFlow pricing tiers…", source="product_overview.md", score=0.9)
            ],
            retrieval_mode="hybrid",
            request_id="test-123",
        )

    def test_query_returns_200(self, client, mock_retriever):
        with patch("src.api.routes.handle", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = self._mock_response()
            r = client.post("/query", json={"question": "What are the pricing tiers?"})
        assert r.status_code == 200

    def test_query_returns_answer(self, client):
        with patch("src.api.routes.handle", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = self._mock_response()
            r = client.post("/query", json={"question": "What are the pricing tiers?"})
        body = r.json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    def test_query_returns_citations(self, client):
        with patch("src.api.routes.handle", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = self._mock_response()
            r = client.post("/query", json={"question": "What are the pricing tiers?"})
        body = r.json()
        assert "citations" in body
        assert isinstance(body["citations"], list)

    def test_query_missing_question_returns_422(self, client):
        r = client.post("/query", json={})
        assert r.status_code == 422

    def test_query_respects_top_k(self, client):
        with patch("src.api.routes.handle", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = self._mock_response()
            r = client.post("/query", json={"question": "test", "top_k": 3})
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# /ingest
# ─────────────────────────────────────────────────────────────────────────────

class TestIngest:
    def test_ingest_returns_200(self, client):
        with patch("src.api.routes.ingest_text") as mock_ingest:
            mock_ingest.return_value = IngestionResult(
                documents_processed=1,
                documents_skipped=0,
                chunks_created=4,
                errors=[],
            )
            r = client.post(
                "/ingest",
                params={"text": "New document content.", "title": "Test"},
            )
        assert r.status_code == 200

    def test_ingest_returns_chunks_created(self, client):
        with patch("src.api.routes.ingest_text") as mock_ingest:
            mock_ingest.return_value = IngestionResult(
                documents_processed=1,
                documents_skipped=0,
                chunks_created=7,
                errors=[],
            )
            r = client.post(
                "/ingest",
                params={"text": "Some text content.", "title": "Doc"},
            )
        body = r.json()
        assert "chunks_created" in body

    def test_ingest_calls_rebuild_bm25(self, client, mock_retriever):
        mock_retriever.rebuild_bm25.reset_mock()
        with patch("src.api.routes.ingest_text") as mock_ingest:
            mock_ingest.return_value = IngestionResult(
                documents_processed=1, documents_skipped=0, chunks_created=2, errors=[],
            )
            client.post("/ingest", params={"text": "Rebuild test.", "title": "X"})
        mock_retriever.rebuild_bm25.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────

class TestMiddleware:
    def test_response_has_request_id_header(self, client):
        r = client.get("/health")
        assert "X-Request-ID" in r.headers

    def test_response_has_latency_header(self, client):
        r = client.get("/health")
        assert "X-Latency-Ms" in r.headers
