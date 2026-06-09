"""
Project 42 — Prompt Version Control System
============================================
Build a production-grade prompt registry with:
  - SQLite-backed versioning (create, promote, rollback, diff)
  - A/B routing (deterministic hash-based split)
  - Canary deployment (10% → 50% → 100% auto-promote)
  - Per-version metrics tracking
  - FastAPI management API

Guide: guide/14_llmops.md §§3, 7
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# ─────────────────────────────────────────────────
# Sample prompts for testing
# ─────────────────────────────────────────────────
PROMPT_V1 = """You are a compliance assistant. Answer questions about GDPR."""

PROMPT_V2 = """You are a senior compliance attorney with 20 years of experience.
Always cite the specific article number when answering.
Keep answers under 150 words unless complexity requires more.
If uncertain, say so explicitly rather than guessing."""

PROMPT_V3 = """You are a senior compliance attorney specializing in EU data protection law.
Always cite the specific GDPR article number when answering.
Structure your response: [Article Reference] [Plain explanation] [Practical implication].
Keep answers under 200 words. If uncertain, say "I recommend consulting a qualified attorney for..."
Always consider cross-border implications for multinational companies."""

TEST_QUERIES = [
    "What does Article 28 of GDPR require?",
    "When is a Data Protection Impact Assessment required?",
    "What is the right to erasure?",
    "How long can personal data be retained?",
    "What constitutes legitimate interest under GDPR?",
]


# ─────────────────────────────────────────────────
# TODO 1: PromptVersion Pydantic model
# ─────────────────────────────────────────────────
# Fields:
#   name: str
#   version: str              e.g. "v1", "v2"
#   stage: str                "development" | "staging" | "production" | "archived"
#   text: str                 full prompt text
#   model: str                e.g. "gpt-4o-mini"
#   temperature: float        0.0–2.0
#   max_tokens: int           512
#   created_at: str           ISO timestamp
#   author: str               ""
#   commit_sha: str           ""
#   hash: str                 SHA-256[:16] of text
#   changelog: str            human-readable summary of change
#   metrics: dict             {"accuracy": 0.0, "quality_avg": 0.0,
#                              "latency_p95_ms": 0.0, "cost_per_call_usd": 0.0,
#                              "total_calls": 0}
#
# Add a validator that auto-computes hash from text if not provided.

class PromptVersion(BaseModel):
    # TODO 1a: Define all fields with appropriate defaults
    pass


# ─────────────────────────────────────────────────
# TODO 2: PromptRegistry
# ─────────────────────────────────────────────────
# SQLite-backed prompt registry. See guide §14.3.3 for reference implementation.
# Extend it with all methods listed below.

class PromptRegistry:
    def __init__(self, db_path: str = "prompt_registry.db"):
        # TODO 2a: Connect to SQLite, create prompts table
        # Schema: id, name, version, stage, text, model, temperature, max_tokens,
        #         created_at, author, commit_sha, hash, changelog, metrics (JSON text)
        # Add UNIQUE(name, version) constraint
        raise NotImplementedError

    def create(
        self,
        name: str,
        text: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 512,
        author: str = "",
        commit_sha: str = "",
        changelog: str = "",
    ) -> str:
        """Create a new version (auto-numbered v1, v2, ...). Returns version string."""
        # TODO 2b: Count existing versions for this name → compute next version
        # Insert row with stage="development", hash=SHA256[:16] of text
        # Return version string (e.g. "v3")
        raise NotImplementedError

    def get(self, name: str, version: str = "latest") -> PromptVersion:
        """Fetch a prompt. version='latest' returns highest id for this name."""
        # TODO 2c: Query DB, return PromptVersion (raise KeyError if not found)
        raise NotImplementedError

    def get_by_stage(self, name: str, stage: str) -> PromptVersion:
        """Fetch the version currently in a given stage."""
        # TODO 2d: Query WHERE name=? AND stage=? ORDER BY id DESC LIMIT 1
        raise NotImplementedError

    def list_versions(self, name: str) -> list[PromptVersion]:
        """Return all versions for a prompt, newest first."""
        # TODO 2e: Query all rows WHERE name=?, ORDER BY id DESC
        raise NotImplementedError

    def promote(self, name: str, version: str, to_stage: str) -> None:
        """Promote version to to_stage. Archive current holder of that stage."""
        # TODO 2f: UPDATE SET stage='archived' WHERE name=? AND stage=to_stage
        #          UPDATE SET stage=to_stage WHERE name=? AND version=?
        raise NotImplementedError

    def rollback(self, name: str) -> str:
        """Roll back production to the most recent archived version. Returns version."""
        # TODO 2g: Find production version → find most recent archived before it → promote it
        raise NotImplementedError

    def update_metrics(self, name: str, version: str, metrics: dict) -> None:
        """Merge metrics into existing metrics dict for this version."""
        # TODO 2h: Load existing metrics, update with new values, save back
        raise NotImplementedError

    def diff(self, name: str, v1: str, v2: str) -> str:
        """Return unified diff of two prompt versions."""
        # TODO 2i: Use difflib.unified_diff on text lines
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 3: ABRouter
# ─────────────────────────────────────────────────
# Deterministic 50/50 traffic split between two prompt versions.
# Same user_id always gets the same variant.

class ABRouter:
    def __init__(self, registry: PromptRegistry):
        # TODO 3a: Store registry, initialise call_counts dict {version: int}
        raise NotImplementedError

    def route(
        self, user_id: str, name: str, v_a: str, v_b: str, split: float = 0.5
    ) -> PromptVersion:
        """Route user to variant A or B deterministically. split=0.5 means 50/50."""
        # TODO 3b: hash user_id → int → mod 100 → compare to split*100
        # Increment call count for chosen variant, return PromptVersion
        raise NotImplementedError

    def stats(self) -> dict[str, int]:
        """Return call counts per variant."""
        # TODO 3c: Return self.call_counts
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 4: CanaryRouter
# ─────────────────────────────────────────────────
# Gradual rollout: 10% → 50% → 100%

class CanaryRouter:
    STAGES = [0.10, 0.50, 1.00]  # canary percentages
    MIN_CALLS_PER_STAGE = 50     # calls to collect before evaluating

    def __init__(self, registry: PromptRegistry):
        # TODO 4a: Store registry, active canary state:
        # {name → {canary_version, current_pct, stage_idx, call_count, metrics}}
        raise NotImplementedError

    def start_canary(self, name: str, canary_version: str) -> None:
        """Begin canary deployment at 10%."""
        # TODO 4b: Set active canary for name at 10%
        raise NotImplementedError

    def route(self, user_id: str, name: str) -> PromptVersion:
        """Route to canary or production based on active canary state."""
        # TODO 4c: If no active canary → return production version
        # If active canary → hash user_id → assign to canary or production by pct
        raise NotImplementedError

    def record_call(self, name: str, version: str, quality: float, latency_ms: float) -> None:
        """Record a call result. Check if we should advance the canary stage."""
        # TODO 4d: If this is the canary version → increment call_count
        # If call_count >= MIN_CALLS_PER_STAGE → evaluate metrics
        # If metrics OK (quality ≥ 7.5, latency ≤ 5000ms) → advance to next stage
        # If metrics fail → rollback (call registry.rollback(name))
        # If final stage (100%) reached → complete canary (registry.promote to production)
        raise NotImplementedError

    def status(self, name: str) -> dict:
        """Return current canary status for a prompt."""
        # TODO 4e: Return active canary state or {"active": False}
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 5: MetricsCollector
# ─────────────────────────────────────────────────
# Track per-version metrics in SQLite.

class MetricsCollector:
    def __init__(self, db_path: str = "prompt_metrics.db"):
        # TODO 5a: Create metrics table
        # Schema: id, name, version, timestamp, latency_ms, cost_usd,
        #         quality_score (NULLABLE), error (BOOLEAN)
        raise NotImplementedError

    def record(
        self,
        name: str,
        version: str,
        latency_ms: float,
        cost_usd: float,
        quality_score: float | None = None,
        error: bool = False,
    ) -> None:
        # TODO 5b: Insert row
        raise NotImplementedError

    def get_report(self, name: str, hours: int = 24) -> dict[str, dict]:
        """Return aggregated metrics per version for the last N hours."""
        # TODO 5c: For each version: count, avg_latency, p95_latency, avg_cost,
        #          avg_quality, error_rate. Return {version: metrics_dict}
        raise NotImplementedError

    def compare(self, name: str, v1: str, v2: str) -> dict:
        """Side-by-side comparison. Identify winner per metric."""
        # TODO 5d: Get metrics for both versions, compare each dimension,
        # Return {"v1": {...}, "v2": {...}, "winner": {metric: version}}
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 6: MonitoredRouter
# ─────────────────────────────────────────────────
# Combines registry + routing + metrics into one query interface.

class MonitoredRouter:
    def __init__(
        self,
        registry: PromptRegistry,
        metrics: MetricsCollector,
        model: str = "gpt-4o-mini",
    ):
        # TODO 6a: Store components, create ABRouter and CanaryRouter
        raise NotImplementedError

    def query(
        self,
        user_query: str,
        user_id: str,
        prompt_name: str,
        agent_fn: Callable[[str, str], str],
        ab_versions: tuple[str, str] | None = None,
    ) -> dict:
        """
        Route query to correct prompt version, call agent_fn, record metrics.
        
        agent_fn signature: (system_prompt: str, user_query: str) → str
        ab_versions: if set, use ABRouter instead of production prompt
        
        Returns: {answer, prompt_version, latency_ms, cost_usd}
        """
        # TODO 6b: Route to correct prompt version (A/B or canary or production)
        # Time the call, compute cost (~$0.003 per call mock if no real API)
        # Record metrics. Update registry metrics every 100 calls.
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 7: FastAPI Application
# ─────────────────────────────────────────────────
app = FastAPI(title="Prompt Registry API", version="1.0.0")

# TODO 7a: Create global registry, metrics, router instances

class CreatePromptRequest(BaseModel):
    text: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    author: str = ""
    changelog: str = ""

class PromoteRequest(BaseModel):
    version: str
    to_stage: str

class CanaryRequest(BaseModel):
    version: str

# TODO 7b: GET /prompts/{name} — return current production prompt
# TODO 7c: GET /prompts/{name}/versions — all versions with metrics
# TODO 7d: POST /prompts/{name} — create new version
# TODO 7e: POST /prompts/{name}/promote — promote to stage
# TODO 7f: POST /prompts/{name}/rollback — rollback to previous production
# TODO 7g: GET /prompts/{name}/diff?v1=v1&v2=v2 — unified diff
# TODO 7h: GET /prompts/{name}/compare?v1=v1&v2=v2 — metric comparison
# TODO 7i: POST /prompts/{name}/canary — start canary deployment
# TODO 7j: GET /prompts/{name}/metrics — metrics report


# ─────────────────────────────────────────────────
# TODO 8: Demo runner
# ─────────────────────────────────────────────────

def simple_agent(system_prompt: str, user_query: str) -> str:
    """Simple litellm call for demo purposes."""
    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content


def run_demo():
    print("=== Prompt Registry Demo ===\n")

    registry = PromptRegistry(db_path=":memory:")
    metrics = MetricsCollector(db_path=":memory:")

    # TODO 8a: Create v1, v2, v3 from PROMPT_V1, PROMPT_V2, PROMPT_V3
    # (author="demo", changelog describes what changed)

    # TODO 8b: Promote v2 to staging, then production
    # Print: "Promoted v2 → staging" and "Promoted v2 → production"

    # TODO 8c: Show diff v1 → v2

    # TODO 8d: A/B test: route 100 queries between v1 and v2
    # Print stats: calls per version, avg quality, avg latency

    # TODO 8e: Start canary for v3 at 10%
    # Send 200 queries, let auto-promote advance through stages

    # TODO 8f: Demonstrate rollback
    # registry.rollback("compliance") → print result

    # TODO 8g: Print final version list with stages
    print("\n=== Final version list ===")
    for pv in registry.list_versions("compliance"):
        print(f"  {pv.version}: stage={pv.stage} | calls={pv.metrics.get('total_calls', 0)}")


if __name__ == "__main__":
    import sys
    if "--server" in sys.argv:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8001)
    else:
        run_demo()
