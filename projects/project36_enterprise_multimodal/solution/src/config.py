"""
solution/src/config.py — Full implementation.
"""
from __future__ import annotations
from functools import lru_cache
from pydantic import Field  # type: ignore
from pydantic_settings import BaseSettings  # type: ignore


class Config(BaseSettings):
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    llm_primary: str = Field("openai/gpt-4o-mini", env="LLM_PRIMARY")
    llm_fallback: str = Field("openai/gpt-3.5-turbo", env="LLM_FALLBACK")
    llm_fine_tuned: str = Field("", env="LLM_FINE_TUNED")
    llm_vision: str = Field("openai/gpt-4o", env="LLM_VISION")

    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")

    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    mem0_api_key: str = Field("", env="MEM0_API_KEY")
    use_llama_guard: bool = Field(False, env="USE_LLAMA_GUARD")
    api_rate_limit_rpm: int = Field(60, env="API_RATE_LIMIT_RPM")

    @property
    def llm_model_chain(self) -> list[str]:
        chain = [self.llm_primary, self.llm_fallback]
        if self.llm_fine_tuned:
            chain.append(self.llm_fine_tuned)
        return chain

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
