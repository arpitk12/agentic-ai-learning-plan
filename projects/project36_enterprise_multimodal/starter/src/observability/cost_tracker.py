"""
src/observability/cost_tracker.py — Per-call token and USD cost tracking.

TODOs:
  1. implement CostTracker class with record(), get_summary(), get_total()
"""
from __future__ import annotations
import sqlite3
import time
import uuid

# USD cost per 1K tokens for common models (input, output)
COST_PER_1K: dict[str, tuple[float, float]] = {
    "openai/gpt-4o":           (0.005,    0.015),
    "openai/gpt-4o-mini":      (0.00015,  0.0006),
    "openai/gpt-3.5-turbo":    (0.0005,   0.0015),
    "openai/gpt-4-turbo":      (0.01,     0.03),
}


# ── TODO 1: CostTracker class ─────────────────────────────────────────────────
# class CostTracker:
#     """SQLite-backed per-call cost tracker."""
#
#     def __init__(self, db_path: str = "/tmp/agent_costs.db"):
#         self._conn = sqlite3.connect(db_path, check_same_thread=False)
#         self._conn.execute("""
#             CREATE TABLE IF NOT EXISTS calls (
#                 id TEXT PRIMARY KEY,
#                 user_id TEXT,
#                 model TEXT,
#                 input_tokens INTEGER,
#                 output_tokens INTEGER,
#                 cost_usd REAL,
#                 ts REAL DEFAULT (unixepoch())
#             )
#         """)
#         self._conn.commit()
#
#     def record(self, user_id: str, model: str,
#                input_tokens: int, output_tokens: int) -> float:
#         """
#         Record one LLM call and return the USD cost.
#
#         Steps:
#           1a. Look up rates in COST_PER_1K (default 0.00015/0.0006 if unknown)
#           1b. cost = (input_tokens/1000 * in_rate) + (output_tokens/1000 * out_rate)
#           1c. INSERT into calls table
#           1d. Return cost
#         """
#         raise NotImplementedError
#
#     def get_summary(self, user_id: str | None = None,
#                     since_hours: float = 24.0) -> dict:
#         """
#         Aggregate cost and token usage.
#
#         Steps:
#           1a. Build WHERE clause: user_id filter + ts > (now - since_hours * 3600)
#           1b. SELECT SUM(cost_usd), SUM(input_tokens), SUM(output_tokens), COUNT(*)
#           1c. Return {total_cost, total_input_tokens, total_output_tokens, calls}
#         """
#         raise NotImplementedError
#
#     def get_total(self, user_id: str | None = None) -> float:
#         """Return total USD cost for a user (or all users if None)."""
#         raise NotImplementedError

raise NotImplementedError("Implement CostTracker in src/observability/cost_tracker.py")
