"""
src/config.py — Centralised configuration loaded from .env

TODOs:
  1. Define a Config class using pydantic_settings.BaseSettings
  2. Add all required environment variables with types + defaults
  3. Add a @cached_property (or module-level singleton) so the app only
     reads the .env file once
"""
from __future__ import annotations
from functools import lru_cache


# ── TODO 1: Import BaseSettings and Field ─────────────────────────────────────
# from pydantic_settings import BaseSettings
# from pydantic import Field


# ── TODO 2: Define the Config class ──────────────────────────────────────────
# class Config(BaseSettings):
#     """
#     All settings are loaded from environment variables / .env file.
#     """
#
#     # LLM
#     openai_api_key: str = Field(..., env="OPENAI_API_KEY")
#     llm_primary: str = Field("openai/gpt-4o-mini", env="LLM_PRIMARY")
#     llm_fallback: str = Field("openai/gpt-3.5-turbo", env="LLM_FALLBACK")
#     llm_fine_tuned: str = Field("", env="LLM_FINE_TUNED")
#     llm_vision: str = Field("openai/gpt-4o", env="LLM_VISION")
#
#     # Neo4j
#     neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
#     neo4j_user: str = Field("neo4j", env="NEO4J_USER")
#     neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
#
#     # Redis
#     redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
#
#     # ChromaDB
#     chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
#
#     # Mem0
#     mem0_api_key: str = Field("", env="MEM0_API_KEY")
#
#     # Guardrails
#     use_llama_guard: bool = Field(False, env="USE_LLAMA_GUARD")
#
#     # Rate limiting
#     api_rate_limit_rpm: int = Field(60, env="API_RATE_LIMIT_RPM")
#
#     # Derived helpers
#     @property
#     def llm_model_chain(self) -> list[str]:
#         """Ordered fallback chain — skip empty fine-tuned slot."""
#         chain = [self.llm_primary, self.llm_fallback]
#         if self.llm_fine_tuned:
#             chain.append(self.llm_fine_tuned)
#         return chain
#
#     model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# ── TODO 3: Create a cached singleton ────────────────────────────────────────
# @lru_cache(maxsize=1)
# def get_config() -> Config:
#     return Config()

raise NotImplementedError("Implement Config in src/config.py")
