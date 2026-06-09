"""
Project 41 — LLMOps Monitoring Hub
===================================
Build a production-grade monitoring hub that wraps any LLM agent with:
  - Full trace capture (SQLite)
  - Embedding drift detection (KS test)
  - Quality drift monitoring (LLM-judge sampling)
  - SLO tracking (Prometheus)
  - Alert manager (Slack / webhook)
  - FastAPI dashboard

Guide: guide/14_llmops.md §§6, 8, 10
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response
import litellm

load_dotenv()

# ─────────────────────────────────────────────────
# Sample data for testing (50 compliance queries)
# ─────────────────────────────────────────────────
REFERENCE_QUERIES = [
    "What does Article 28 of GDPR require for data processors?",
    "Explain the right to erasure under GDPR Article 17.",
    "What is a Data Processing Agreement and when is it required?",
    "Does GDPR apply to companies outside the EU?",
    "What are the penalties for GDPR non-compliance?",
    "What personal data categories require explicit consent?",
    "Explain legitimate interest as a lawful basis under GDPR.",
    "What is the difference between a data controller and a processor?",
    "How long can personal data be retained under GDPR?",
    "What rights do data subjects have under GDPR?",
] * 5  # 50 reference queries

DRIFTED_QUERIES = [
    "Write me a poem about the ocean.",
    "What is the capital of France?",
    "How do I make pasta carbonara?",
    "Explain quantum entanglement.",
    "What are the best hiking trails in Colorado?",
    "How does photosynthesis work?",
    "What is the plot of Hamlet?",
    "How do I train a machine learning model?",
    "What is blockchain technology?",
    "Recommend a good science fiction novel.",
] * 15  # 150 drifted queries


# ─────────────────────────────────────────────────
# TODO 1: TraceLogger
# ─────────────────────────────────────────────────
# Build a SQLite-backed trace logger.
# Schema: id (auto), trace_id (TEXT), timestamp (TEXT), model (TEXT),
#         prompt_name (TEXT), prompt_version (TEXT), user_id (TEXT),
#         tokens_in (INT), tokens_out (INT), latency_ms (REAL),
#         cost_usd (REAL), error (TEXT NULLABLE), quality_score (REAL NULLABLE).
#
# Methods:
#   log(trace: CallTrace) → None
#   get_recent(n: int = 100) → list[dict]
#   get_by_user(user_id: str) → list[dict]
#   get_stats(hours: int = 24) → dict  # avg latency, total cost, error count, call count

class TraceLogger:
    def __init__(self, db_path: str = "traces.db"):
        # TODO 1a: Connect to SQLite and create table if not exists
        raise NotImplementedError

    def log(self, trace: "CallTrace") -> None:
        # TODO 1b: Insert trace into DB
        raise NotImplementedError

    def get_recent(self, n: int = 100) -> list[dict]:
        # TODO 1c: Return last n traces as list of dicts
        raise NotImplementedError

    def get_stats(self, hours: int = 24) -> dict:
        # TODO 1d: Return {avg_latency_ms, total_cost_usd, error_count, call_count,
        #                   avg_quality_score, p95_latency_ms}
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 2: CostCalculator
# ─────────────────────────────────────────────────
# Map model names to per-token prices and compute cost from a litellm response.
# Prices (per million tokens):
#   gpt-4o-mini:      in=$0.15,   out=$0.60
#   gpt-4o:           in=$2.50,   out=$10.00
#   gpt-4.1:          in=$2.00,   out=$8.00
#   claude-haiku-4:   in=$0.80,   out=$4.00
#   claude-sonnet-4:  in=$3.00,   out=$15.00
#   claude-opus-4:    in=$15.00,  out=$75.00
#
# Method: calculate(model: str, tokens_in: int, tokens_out: int) → float

class CostCalculator:
    PRICES: dict[str, dict[str, float]] = {
        # TODO 2a: Fill in the price table (cost per million tokens, not per token)
    }

    def calculate(self, model: str, tokens_in: int, tokens_out: int) -> float:
        # TODO 2b: Look up model prices (use "gpt-4o-mini" as default if model unknown)
        # Return cost_usd = tokens_in * in_price / 1_000_000 + tokens_out * out_price / 1_000_000
        raise NotImplementedError


# ─────────────────────────────────────────────────
# CallTrace dataclass (provided — do not modify)
# ─────────────────────────────────────────────────
@dataclass
class CallTrace:
    trace_id: str
    model: str
    prompt_name: str
    prompt_version: str
    user_id: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    quality_score: float | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────
# TODO 3: EmbeddingDriftDetector
# ─────────────────────────────────────────────────
# Detect when input queries shift semantically from a reference distribution.
# Algorithm: KS test on PCA projection of embeddings (see guide §14.6.3).
#
# Phase 1 (first reference_size queries): build reference set
# Phase 2 (after): check every check_interval queries
#
# Use a simple embedding approximation (TF-IDF for testing, real embeddings in prod):
#   from sklearn.feature_extraction.text import TfidfVectorizer
#   (You can also use litellm.embedding() with text-embedding-3-small)
#
# Method: add(text: str) → None
# Method: check() → dict  # {drift, ks_stat, p_value, reason, n_reference, n_current}

class EmbeddingDriftDetector:
    def __init__(
        self,
        reference_size: int = 50,
        window_size: int = 50,
        check_interval: int = 25,
        significance: float = 0.05,
    ):
        # TODO 3a: Initialise vectorizer, reference list, current window, locked flag
        raise NotImplementedError

    def add(self, text: str) -> None:
        # TODO 3b: Add text to reference (if not locked) or current window (if locked)
        # When reference reaches reference_size, fit vectorizer and lock
        raise NotImplementedError

    def _project(self, texts: list[str]) -> np.ndarray:
        # TODO 3c: Transform texts → TF-IDF → PCA first component → 1D array
        raise NotImplementedError

    def check(self) -> dict:
        # TODO 3d: Run ks_2samp on projections of reference vs current window
        # Return {drift, ks_stat, p_value, reason, n_reference, n_current}
        # If not enough data, return {drift: False, reason: "building reference"}
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 4: QualityDriftMonitor
# ─────────────────────────────────────────────────
# Sample a fraction of production traffic and score it with an LLM judge.
# Track rolling average; alert when it drops vs baseline.
#
# LLM judge prompt: "Rate the following response on a scale of 1-10 for accuracy
# and helpfulness. Respond with only a number.\nQuery: {query}\nResponse: {response}"
#
# Methods:
#   maybe_score(query, response) → float | None  (returns score only if sampled)
#   check() → dict  # {drift, current_avg, baseline, drop, sample_count, reason}

class QualityDriftMonitor:
    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        sample_rate: float = 0.05,
        window_size: int = 50,
        drop_threshold: float = 0.5,
    ):
        # TODO 4a: Initialise scores deque, baseline, judge_model, sample_rate
        raise NotImplementedError

    def _llm_judge(self, query: str, response: str) -> float:
        # TODO 4b: Call litellm with judge prompt, parse score from response (1.0–10.0)
        # Return 7.0 as default if parsing fails
        raise NotImplementedError

    def maybe_score(self, query: str, response: str) -> float | None:
        # TODO 4c: random.random() < sample_rate → call _llm_judge, append to scores
        raise NotImplementedError

    def check(self) -> dict:
        # TODO 4d: Return drift status vs baseline
        # Set baseline on first check (when >= 5 scores); detect drop > drop_threshold
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 5: SLOTracker
# ─────────────────────────────────────────────────
# Emit Prometheus metrics for all SLO dimensions.
# Initialise these prometheus_client objects at class level:
#
#   REQUEST_LATENCY = Histogram("llm_request_duration_seconds", ...,
#                               buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
#   REQUEST_ERRORS  = Counter("llm_request_errors_total", ..., labelnames=["error_type"])
#   QUALITY_SCORE   = Gauge("llm_quality_score", ...)
#   COST_PER_CALL   = Histogram("llm_cost_per_call_usd", ...,
#                               buckets=[0.001, 0.005, 0.01, 0.02, 0.05])
#   TOKENS_TOTAL    = Counter("llm_tokens_total", ..., labelnames=["direction"])
#
# Method: track(latency_s, cost_usd, tokens_in, tokens_out, error=None, quality=None)
# Method: slo_status() → dict  # summary of current SLO health (pass/warn/fail per metric)

class SLOTracker:
    # TODO 5a: Define Prometheus metrics as class-level variables (outside __init__)

    SLO_LIMITS = {
        "p95_latency_s": 5.0,
        "error_rate_pct": 0.5,
        "quality_score_min": 7.5,
        "cost_per_call_usd_max": 0.015,
    }

    def __init__(self):
        # TODO 5b: Initialise rolling buffer for latency (last 100) and error tracking
        raise NotImplementedError

    def track(
        self,
        latency_s: float,
        cost_usd: float,
        tokens_in: int,
        tokens_out: int,
        error: str | None = None,
        quality: float | None = None,
    ) -> None:
        # TODO 5c: Observe all Prometheus metrics
        raise NotImplementedError

    def slo_status(self) -> dict:
        # TODO 5d: Return dict with status per SLO dimension
        # {"p95_latency": {"value": 1.8, "limit": 5.0, "status": "OK"},
        #  "error_rate": {"value": 0.0, "limit": 0.5, "status": "OK"}, ...}
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 6: AlertManager
# ─────────────────────────────────────────────────
# Rule-based alert engine with deduplication (same alert silent for 30 min).
# Alert rules (check these in order):
#   - quality drop > 0.5 pts  → WARNING
#   - embedding drift          → WARNING
#   - error rate > 1%          → CRITICAL
#   - P95 latency > 8s         → CRITICAL
#   - daily cost > $10         → WARNING
#
# Method: check_and_alert(slo_status, drift_status, quality_status) → list[str]
#   Returns list of alert messages fired this check.
#
# Method: send_slack(message: str) → None
#   POST to SLACK_WEBHOOK_URL env var (skip if not set, just print)
#
# Method: get_alert_history(n: int = 50) → list[dict]

class AlertManager:
    COOLDOWN_MINUTES = 30

    def __init__(self, db_path: str = "alerts.db"):
        # TODO 6a: Connect to SQLite, create alerts table
        # Schema: id, timestamp, severity, message
        raise NotImplementedError

    def check_and_alert(
        self,
        slo_status: dict,
        drift_status: dict,
        quality_status: dict,
    ) -> list[str]:
        # TODO 6b: Evaluate all rules, fire alerts not in cooldown
        # Return list of alert messages that were fired
        raise NotImplementedError

    def _fire(self, severity: str, message: str) -> bool:
        # TODO 6c: Check cooldown (same message in last COOLDOWN_MINUTES → skip)
        # If not in cooldown: insert to DB, call send_slack, return True
        raise NotImplementedError

    def send_slack(self, message: str) -> None:
        # TODO 6d: POST {"text": message} to SLACK_WEBHOOK_URL env var
        # If env var not set, just print the message
        raise NotImplementedError

    def get_alert_history(self, n: int = 50) -> list[dict]:
        # TODO 6e: Return last n alerts from DB
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 7: MonitoredAgent
# ─────────────────────────────────────────────────
# Orchestrate all components into a single query method.
#
# Constructor params: model, system_prompt, prompt_name, prompt_version
# Method: query(user_query: str, user_id: str) → dict
#   Returns: {answer, trace_id, latency_ms, cost_usd, tokens_in, tokens_out}
#
# Behaviour:
#   1. Call litellm.completion() and time it
#   2. Calculate cost with CostCalculator
#   3. Log trace with TraceLogger
#   4. Update SLO metrics with SLOTracker
#   5. Add query to EmbeddingDriftDetector
#   6. Maybe score with QualityDriftMonitor
#   7. Every 25 calls: run all drift checks + AlertManager.check_and_alert()

class MonitoredAgent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful compliance assistant.",
        prompt_name: str = "compliance-agent",
        prompt_version: str = "v1",
    ):
        # TODO 7a: Instantiate all components (TraceLogger, CostCalculator,
        #          EmbeddingDriftDetector, QualityDriftMonitor, SLOTracker, AlertManager)
        raise NotImplementedError

    def query(self, user_query: str, user_id: str = "anonymous") -> dict:
        # TODO 7b: Implement the full monitored call (see steps 1-7 above)
        raise NotImplementedError

    def _run_periodic_checks(self) -> None:
        # TODO 7c: Run drift checks + alert manager + print status summary
        raise NotImplementedError


# ─────────────────────────────────────────────────
# TODO 8: FastAPI Application
# ─────────────────────────────────────────────────
# Build a FastAPI app with these endpoints:
#   POST /query          → MonitoredAgent.query()
#   GET  /metrics        → Prometheus format (use generate_latest())
#   GET  /dashboard      → JSON with last-24h stats + SLO status + drift status
#   GET  /drift          → Current drift status from all detectors
#   GET  /alerts         → Last 50 alerts
#   GET  /traces         → Last 100 traces (query param: n=100)

app = FastAPI(title="LLMOps Monitoring Hub")

# TODO 8a: Create a global MonitoredAgent instance

class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"

# TODO 8b: Implement POST /query endpoint
# TODO 8c: Implement GET /metrics endpoint (return Prometheus text format)
# TODO 8d: Implement GET /dashboard endpoint
# TODO 8e: Implement GET /drift endpoint
# TODO 8f: Implement GET /alerts endpoint
# TODO 8g: Implement GET /traces endpoint


# ─────────────────────────────────────────────────
# TODO 9: Simulation Runner
# ─────────────────────────────────────────────────
# Send 50 reference queries, then 150 drifted queries.
# Verify drift is detected after the distribution shift.
# Print a summary at the end.

def simulate_traffic(agent: MonitoredAgent, n_reference: int = 50, n_drifted: int = 150):
    print("=== LLMOps Monitoring Hub — Traffic Simulation ===\n")
    
    # Phase 1: Reference traffic (compliance queries)
    print(f"Phase 1: Sending {n_reference} reference queries...")
    for i, query in enumerate(REFERENCE_QUERIES[:n_reference]):
        # TODO 9a: Send query with user_id = f"user-{i % 10}"
        # Print progress every 10 queries
        pass

    print(f"\nPhase 2: Sending {n_drifted} drifted queries (topic shift)...")
    for i, query in enumerate(DRIFTED_QUERIES[:n_drifted]):
        # TODO 9b: Send query with user_id = f"user-{i % 10}"
        # Print progress every 25 queries
        pass

    # TODO 9c: Print final summary
    # - Total calls, total cost, avg latency
    # - Drift status (was drift detected?)
    # - Alert history (how many alerts fired?)
    # - SLO status table


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--server" in sys.argv:
        import uvicorn
        print("Starting LLMOps Monitoring Hub API at http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run simulation
        agent = MonitoredAgent(
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant. Answer concisely.",
            prompt_name="demo-agent",
            prompt_version="v1",
        )
        simulate_traffic(agent, n_reference=50, n_drifted=150)
