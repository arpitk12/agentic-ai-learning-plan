"""
src/resilience/circuit_breaker.py — Circuit breaker state machine.

TODOs:
  1. Define CircuitState enum
  2. Implement CircuitBreaker class with state property, can_attempt(),
     record_success(), record_failure()
"""
from __future__ import annotations
import time
from enum import Enum


# ── TODO 1: State enum ────────────────────────────────────────────────────────
# class CircuitState(Enum):
#     CLOSED    = "closed"     # normal — requests pass through
#     OPEN      = "open"       # tripped — fail fast without calling model
#     HALF_OPEN = "half_open"  # recovery probe — allow one test request


# ── TODO 2: CircuitBreaker class ─────────────────────────────────────────────
# class CircuitBreaker:
#     def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 30.0):
#         self.name = name
#         self._threshold = failure_threshold
#         self._timeout = recovery_timeout
#         self._failures = 0
#         self._opened_at: float | None = None
#
#     @property
#     def state(self) -> CircuitState:
#         """
#         CLOSED:    _failures < _threshold
#         OPEN:      _failures >= _threshold AND not enough time has passed
#         HALF_OPEN: _failures >= _threshold AND recovery_timeout has expired
#         """
#         if self._failures < self._threshold:
#             return CircuitState.CLOSED
#         if self._opened_at and (time.time() - self._opened_at) > self._timeout:
#             return CircuitState.HALF_OPEN
#         return CircuitState.OPEN
#
#     def can_attempt(self) -> bool:
#         """Return True when state is CLOSED or HALF_OPEN."""
#         return self.state != CircuitState.OPEN
#
#     def record_success(self):
#         """Reset failure counter and clear opened_at (move back to CLOSED)."""
#         self._failures = 0
#         self._opened_at = None
#
#     def record_failure(self):
#         """Increment failures. If threshold reached, record time of tripping."""
#         self._failures += 1
#         if self._failures >= self._threshold and self._opened_at is None:
#             self._opened_at = time.time()

raise NotImplementedError("Implement CircuitBreaker in src/resilience/circuit_breaker.py")
