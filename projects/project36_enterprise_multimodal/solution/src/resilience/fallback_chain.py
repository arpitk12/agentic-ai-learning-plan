"""
solution/src/resilience/circuit_breaker.py + fallback_chain.py — Full implementation.
Both classes in one file for convenience; import from here in all solution modules.
"""
from __future__ import annotations
import asyncio
import time
from enum import Enum
import litellm  # type: ignore


class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.name = name
        self._threshold = failure_threshold
        self._timeout = recovery_timeout
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._failures < self._threshold:
            return CircuitState.CLOSED
        if self._opened_at and (time.time() - self._opened_at) > self._timeout:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def can_attempt(self) -> bool:
        return self.state != CircuitState.OPEN

    def record_success(self):
        self._failures = 0
        self._opened_at = None

    def record_failure(self):
        self._failures += 1
        if self._failures >= self._threshold and self._opened_at is None:
            self._opened_at = time.time()
            print(f"  ⚡ Circuit [{self.name}] TRIPPED after {self._failures} failures")


class FallbackChain:
    def __init__(self, models: list[str], failure_threshold: int = 3,
                 recovery_timeout: float = 30.0, max_retries: int = 2):
        self._models = models
        self._breakers = {m: CircuitBreaker(m, failure_threshold, recovery_timeout)
                          for m in models}
        self._max_retries = max_retries

    async def call(self, messages: list[dict], **kwargs) -> tuple[str, str]:
        """Try each model in order. Returns (reply, model_used). Raises if all fail."""
        for model in self._models:
            breaker = self._breakers[model]
            if not breaker.can_attempt():
                continue
            for attempt in range(self._max_retries + 1):
                try:
                    resp = await litellm.acompletion(model=model, messages=messages, **kwargs)
                    breaker.record_success()
                    return resp.choices[0].message.content, model
                except Exception as e:
                    breaker.record_failure()
                    if attempt < self._max_retries:
                        await asyncio.sleep(2 ** attempt)   # 1s, 2s backoff
        raise RuntimeError(f"All models failed: {self._models}")

    def status(self) -> dict[str, str]:
        return {m: b.state.value for m, b in self._breakers.items()}

    def circuits(self) -> list[dict]:
        return [{"model": m, "state": b.state.value, "failures": b._failures}
                for m, b in self._breakers.items()]
