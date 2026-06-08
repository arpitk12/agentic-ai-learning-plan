"""
Centralised configuration — all settings loaded from .env via pydantic-settings.
Import `cfg` in any module; never hardcode values elsewhere.
"""
from __future__ import annotations
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────
    model: str = "gemini/gemini-1.5-flash"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""

    # ── Qdrant ───────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_docs"
    qdrant_vector_size: int = 384
    qdrant_use_grpc: bool = False
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 200
    qdrant_search_ef: int = 128

    # ── Kafka ────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_raw_topic: str = "raw-documents"
    kafka_chunks_topic: str = "document-chunks"
    kafka_embedded_topic: str = "embedded-chunks"
    kafka_dlq_topic: str = "dlq-ingestion"
    kafka_embed_batch_size: int = 32
    kafka_index_batch_size: int = 256

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_query_cache_ttl: int = 3600
    redis_embed_cache_ttl: int = 86400

    # ── Semantic cache ───────────────────────────────────────────────────
    semantic_cache_threshold: float = 0.97
    semantic_cache_size: int = 5000

    # ── Retrieval ────────────────────────────────────────────────────────
    retrieval_top_k: int = 20
    reranker_top_n: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Zero-hallucination ───────────────────────────────────────────────
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    faithfulness_threshold: float = 0.75      # per-sentence entailment
    overall_faithfulness_threshold: float = 0.80  # min fraction grounded
    min_retrieval_score: float = 0.65         # abstain below this

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # ── API ──────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


cfg = Config()
