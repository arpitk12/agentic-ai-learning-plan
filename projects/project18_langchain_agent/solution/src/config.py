"""Configuration for project 18 — LangChain Research Agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    model: str = field(default_factory=lambda: os.getenv("MODEL", "openai/gpt-4o-mini"))
    litellm_api_base: str | None = field(default_factory=lambda: os.getenv("LITELLM_API_BASE"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    langsmith_api_key: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY"))
    langsmith_project: str = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "project18-langchain"))
    faiss_index_path: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


cfg = Config()
