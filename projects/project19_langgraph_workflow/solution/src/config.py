"""Configuration for project 19 — LangGraph Code Review Workflow."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    model: str = field(default_factory=lambda: os.getenv("MODEL", "openai/gpt-4o-mini"))
    litellm_api_base: str | None = field(default_factory=lambda: os.getenv("LITELLM_API_BASE"))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    checkpoint_db: str = field(default_factory=lambda: os.getenv("CHECKPOINT_DB", "data/checkpoints.db"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


cfg = Config()
