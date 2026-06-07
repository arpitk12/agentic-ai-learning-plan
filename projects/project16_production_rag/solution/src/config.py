"""
Central configuration — every module imports `cfg` from here.
All settings driven by environment variables with sensible defaults.
"""
import os
from pathlib import Path


class Config:
    # ── LLM ───────────────────────────────────────────────────────────────
    MODEL: str           = os.getenv("MODEL", "gemini/gemini-2.0-flash")
    LLM_MAX_TOKENS: int  = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # ── Embedding (local — sentence-transformers, no API cost) ─────────────
    EMBED_MODEL: str     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    EMBED_DIM: int       = int(os.getenv("EMBED_DIM", "384"))
    EMBED_BATCH: int     = int(os.getenv("EMBED_BATCH", "64"))

    # ── Vector store ───────────────────────────────────────────────────────
    # Local mode (default): persists to CHROMA_PATH on disk
    # HTTP mode (docker-compose): set CHROMA_HOST to connect to ChromaDB service
    CHROMA_PATH: str     = os.getenv("CHROMA_PATH", "./data/chroma_db")
    CHROMA_HOST: str     = os.getenv("CHROMA_HOST", "")   # empty = local mode
    CHROMA_PORT: int     = int(os.getenv("CHROMA_PORT", "8001"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "prod_rag")

    # ── Chunking ───────────────────────────────────────────────────────────
    CHUNK_SIZE: int      = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int   = int(os.getenv("CHUNK_OVERLAP", "64"))
    MIN_CHUNK_LEN: int   = int(os.getenv("MIN_CHUNK_LEN", "50"))

    # ── Retrieval ──────────────────────────────────────────────────────────
    TOP_K: int           = int(os.getenv("TOP_K", "6"))       # candidates
    RERANK_TOP_N: int    = int(os.getenv("RERANK_TOP_N", "3")) # after rerank
    BM25_WEIGHT: float   = float(os.getenv("BM25_WEIGHT", "0.3"))  # hybrid blend
    VECTOR_WEIGHT: float = 1.0 - float(os.getenv("BM25_WEIGHT", "0.3"))

    # ── API ────────────────────────────────────────────────────────────────
    API_HOST: str        = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int        = int(os.getenv("API_PORT", "8000"))
    RATE_LIMIT: int      = int(os.getenv("RATE_LIMIT", "60"))  # req/min per IP

    # ── Evaluation ─────────────────────────────────────────────────────────
    EVAL_PASS_THRESHOLD: float = float(os.getenv("EVAL_PASS_THRESHOLD", "0.75"))

    # ── Paths ──────────────────────────────────────────────────────────────
    DATA_DIR: Path       = Path(os.getenv("DATA_DIR", "data/sample_docs"))
    CHROMA_DIR: Path     = Path(CHROMA_PATH)

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str       = os.getenv("LOG_LEVEL", "INFO")
    LOG_JSON: bool       = os.getenv("LOG_JSON", "true").lower() == "true"


cfg = Config()
