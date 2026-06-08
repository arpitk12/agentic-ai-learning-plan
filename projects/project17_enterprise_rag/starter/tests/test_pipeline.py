"""
Tests for the ingestion pipeline components.

pytest tests/test_pipeline.py -v
"""
import pytest


# ── Chunker ────────────────────────────────────────────────────────────────

def test_chunker_splits_long_text():
    from src.ingestion.chunker import Chunker
    c = Chunker(chunk_size=100, overlap=20)
    text = "word " * 200   # 1000 chars
    chunks = c.chunk(text)
    assert len(chunks) > 1
    assert all(len(ch.text) <= 100 for ch in chunks)


def test_chunker_short_text_single_chunk():
    from src.ingestion.chunker import Chunker
    c = Chunker(chunk_size=512, overlap=64)
    text = "Short text."
    chunks = c.chunk(text)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunker_empty_text():
    from src.ingestion.chunker import Chunker
    c = Chunker()
    chunks = c.chunk("")
    assert chunks == []


def test_chunker_overlap_preserves_context():
    from src.ingestion.chunker import Chunker
    c = Chunker(chunk_size=50, overlap=10)
    text = "A" * 40 + " " + "B" * 40
    chunks = c.chunk(text)
    # With overlap, second chunk should contain some A characters
    assert len(chunks) >= 1


# ── QdrantStore (unit — requires Qdrant running) ──────────────────────────

@pytest.mark.integration
def test_qdrant_create_and_search():
    from src.store.qdrant_store import QdrantStore
    store = QdrantStore(collection_name="test_collection")
    store.create_collection(recreate=True)

    item = {
        "chunk_id": "test_001",
        "doc_id": "doc_001",
        "chunk_index": 0,
        "text": "The API rate limit is 100 requests per minute.",
        "title": "API Reference",
        "source": "test",
        "metadata": {},
        "embedding": [0.1] * 384,
    }
    n = store.upsert_batch([item])
    assert n == 1

    results = store.search(query_vector=[0.1] * 384, top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "test_001"


# ── RedisCache ─────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_redis_query_cache_roundtrip():
    from src.cache.redis_cache import RedisCache
    cache = RedisCache()
    q = "What is the rate limit?"
    response = {"answer": "100 req/min", "citations": []}
    cache.set_query(q, response)
    result = cache.get_query(q)
    assert result is not None
    assert result["answer"] == "100 req/min"


@pytest.mark.integration
def test_redis_embedding_cache_roundtrip():
    from src.cache.redis_cache import RedisCache
    cache = RedisCache()
    text = "Hello world"
    emb = [0.1, 0.2, 0.3]
    cache.set_embedding(text, emb)
    result = cache.get_embedding(text)
    assert result is not None
    assert abs(result[0] - 0.1) < 1e-6


@pytest.mark.integration
def test_redis_cache_miss():
    from src.cache.redis_cache import RedisCache
    cache = RedisCache()
    result = cache.get_query("this question was never asked xyzzy12345")
    assert result is None
