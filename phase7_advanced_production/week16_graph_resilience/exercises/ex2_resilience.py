"""
Exercise 2: Circuit Breaker + Fallback Chain + Saga Pattern
Phase 7 / Week 16 — Graph RAG · Resilience · A2A · Multi-Tenancy · Reasoning

Goal: Build a production-grade resilience layer for agent systems:
      - Circuit breaker that prevents cascading failures
      - Model fallback chain with escalating fallbacks to local models
      - Saga pattern for multi-step agent actions with automatic rollback

Stack: tenacity · litellm · asyncio (no extra dependencies)

pip install tenacity litellm python-dotenv

TODOs:
  1. Implement CircuitBreaker class (open/half-open/closed states)
  2. Build a model fallback chain with per-model circuit breakers
  3. Implement retry with exponential backoff (tenacity)
  4. Build the Saga coordinator with compensation
  5. Implement a dead letter queue for permanently failed tasks
  6. Build an idempotency layer to safely retry agent runs
  7. BONUS: Health check endpoint for all models in the fallback chain
"""
from __future__ import annotations
import os, json, asyncio, time, hashlib, sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Awaitable, Any
import litellm
from dotenv import load_dotenv

load_dotenv()

# ── TODO 1: Circuit Breaker ───────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "closed"         # normal, requests pass through
    OPEN = "open"             # failing, requests rejected immediately
    HALF_OPEN = "half_open"   # recovery, let ONE request through to test

@dataclass
class CircuitBreaker:
    """
    TODO 1: Implement the circuit breaker pattern.

    State machine:
      CLOSED → OPEN: after failure_threshold consecutive failures
      OPEN → HALF_OPEN: after recovery_timeout seconds
      HALF_OPEN → CLOSED: if the test request succeeds
      HALF_OPEN → OPEN: if the test request fails

    Implement:
    a) Properties:
       - state: CircuitState (computed, not stored)
         CLOSED if _failures < failure_threshold
         OPEN if _opened_at is recent (< recovery_timeout ago)
         HALF_OPEN if _opened_at is old (>= recovery_timeout ago)

    b) Methods:
       - record_success(): reset _failures=0, _opened_at=None, state→CLOSED
       - record_failure(): increment _failures; if >= threshold: _opened_at=now, state→OPEN
       - can_attempt() -> bool: return state != CircuitState.OPEN
         (CLOSED: yes, HALF_OPEN: yes but only for test request, OPEN: no)
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 30      # seconds until OPEN → HALF_OPEN
    _failures: int = field(default=0, init=False)
    _opened_at: datetime | None = field(default=None, init=False)
    _total_requests: int = field(default=0, init=False)
    _total_failures: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        # TODO 1a: implement state computation
        raise NotImplementedError

    def can_attempt(self) -> bool:
        # TODO 1b: return True if request should be allowed through
        raise NotImplementedError

    def record_success(self) -> None:
        # TODO 1c: implement success handling
        raise NotImplementedError

    def record_failure(self) -> None:
        # TODO 1d: implement failure handling
        raise NotImplementedError

    def stats(self) -> dict:
        """Return current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._failures,
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "failure_rate": round(self._total_failures / max(1, self._total_requests), 3),
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
        }

# ── TODO 2: Model Fallback Chain ──────────────────────────────────────────────

FALLBACK_MODELS = [
    "openai/gpt-4o-mini",           # primary (best quality)
    "anthropic/claude-3-haiku-20240307",  # fallback 1
    "groq/llama-3.3-70b-versatile", # fallback 2 (free, fast)
    "ollama/llama3.2",              # fallback 3 (local, always available)
]

class FallbackChain:
    """
    TODO 2: Manage a chain of models with per-model circuit breakers.

    In __init__(self, models=FALLBACK_MODELS):
      - Create a CircuitBreaker for each model:
        self.breakers = {model: CircuitBreaker(name=model) for model in models}
      - self.models = models

    Implement async call(self, messages, **kwargs) -> tuple[str, str]:
      Returns (response_text, model_used).

    Algorithm:
    a) For each model in self.models:
       - breaker = self.breakers[model]
       - if not breaker.can_attempt():
           log f"Circuit OPEN for {model}, skipping"
           continue
       - Try: result = await litellm.acompletion(model=model, messages=messages, **kwargs)
              breaker.record_success()
              return (result.choices[0].message.content, model)
       - Except Exception as e:
              breaker.record_failure()
              log f"Model {model} failed: {e} (state: {breaker.state})"
              continue

    b) If ALL models failed: raise RuntimeError("All models in fallback chain failed")

    Also implement:
    def print_status(self) -> None:
      Print a table showing the state of each model's circuit breaker.
    """
    def __init__(self, models: list[str] = FALLBACK_MODELS):
        # TODO 2: implement here
        raise NotImplementedError

    async def call(self, messages: list[dict], **kwargs) -> tuple[str, str]:
        # TODO 2: implement here
        raise NotImplementedError

    def print_status(self) -> None:
        # TODO 2: implement here
        raise NotImplementedError

# ── TODO 3: Retry with Exponential Backoff ────────────────────────────────────

def make_resilient_caller(chain: FallbackChain):
    """
    TODO 3: Wrap the fallback chain with tenacity retry logic.

    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    Create an async function resilient_call(messages, **kwargs) that:
    a) Is decorated with @retry(
           stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=10),
           retry=retry_if_exception_type((asyncio.TimeoutError, ConnectionError)),
           reraise=True,
       )

    b) Calls: return await asyncio.wait_for(chain.call(messages, **kwargs), timeout=30.0)
       (30-second timeout per attempt)

    c) On retry: log the attempt number and wait time.

    Return the decorated function.
    """
    # TODO 3: implement here
    raise NotImplementedError

# ── TODO 4: Saga Pattern ──────────────────────────────────────────────────────

@dataclass
class SagaStep:
    name: str
    action: Callable[..., Awaitable[dict]]       # forward step
    compensate: Callable[..., Awaitable[None]]   # undo step

class SagaCoordinator:
    """
    TODO 4: Execute a saga — a sequence of steps with automatic rollback.

    The Saga pattern ensures that if step N fails, steps 0..N-1 are compensated
    (rolled back) in reverse order.

    Implement async execute(self, steps: list[SagaStep], context: dict) -> dict:

    a) completed_steps = []
    b) For each step in steps:
       - log f"Executing step: {step.name}"
       - context = await step.action(context)
       - completed_steps.append(step)
    c) If any step raises:
       - log f"Step '{step.name}' failed. Compensating {len(completed_steps)} steps..."
       - For each step in reversed(completed_steps):
           try: await step.compensate(context)
           except Exception as comp_err: log error but continue
       - raise SagaError(f"Saga failed at '{step.name}'") from original_error
    d) Return final context on success.
    """
    async def execute(self, steps: list[SagaStep], context: dict) -> dict:
        # TODO 4: implement here
        raise NotImplementedError

class SagaError(Exception):
    pass

# ── Saga steps for document review ────────────────────────────────────────────

async def reserve_review_slot(ctx: dict) -> dict:
    """Step 1: Reserve a processing slot (simulated)."""
    print(f"   → Reserving slot for {ctx['doc_id']}")
    await asyncio.sleep(0.1)
    ctx["slot_id"] = f"SLOT-{ctx['doc_id']}"
    return ctx

async def release_review_slot(ctx: dict) -> None:
    """Compensate step 1."""
    print(f"   ← Releasing slot {ctx.get('slot_id')}")

async def extract_document(ctx: dict) -> dict:
    """Step 2: Extract and validate document."""
    print(f"   → Extracting document {ctx['doc_id']}")
    await asyncio.sleep(0.1)
    ctx["extracted"] = True
    return ctx

async def delete_extraction(ctx: dict) -> None:
    """Compensate step 2."""
    print(f"   ← Deleting extraction for {ctx['doc_id']}")

async def run_llm_review(ctx: dict, chain: FallbackChain) -> dict:
    """Step 3: Run compliance review via LLM."""
    print(f"   → Running LLM review for {ctx['doc_id']}")
    text, model = await chain.call([{
        "role": "user",
        "content": f"Review document {ctx['doc_id']} for compliance. Return JSON: {{\"risk\": \"low|medium|high\"}}"
    }], max_tokens=50)
    ctx["review_result"] = text
    ctx["model_used"] = model
    return ctx

async def archive_failed_review(ctx: dict) -> None:
    """Compensate step 3."""
    print(f"   ← Archiving failed review for {ctx['doc_id']}")

async def store_to_db(ctx: dict) -> dict:
    """Step 4: Store result to database."""
    print(f"   → Storing result for {ctx['doc_id']}")
    await asyncio.sleep(0.05)
    ctx["db_stored"] = True
    return ctx

async def delete_from_db(ctx: dict) -> None:
    """Compensate step 4."""
    print(f"   ← Deleting DB record for {ctx['doc_id']}")

# ── TODO 5: Dead Letter Queue ─────────────────────────────────────────────────

class DeadLetterQueue:
    """
    TODO 5: Store permanently failed tasks for manual review or retry.

    In __init__(self, db_path="dlq.db"):
      CREATE TABLE dead_letters (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          task_id TEXT UNIQUE,
          task_type TEXT,
          payload TEXT,  -- JSON
          error TEXT,
          attempts INTEGER,
          created_at TEXT,
          last_attempt TEXT
      )

    Implement:
    a) send(self, task_id, task_type, payload, error, attempts=1)
       - INSERT OR REPLACE into dead_letters

    b) get_all(self) -> list[dict]
       - SELECT all rows, parse payload JSON

    c) retry(self, task_id) -> dict | None
       - Get the task, delete it from DLQ, return the payload
       - If not found, return None

    d) print_summary(self) -> None
       - Print count of tasks by task_type
    """
    def __init__(self, db_path: str = "dlq.db"):
        # TODO 5: implement here
        raise NotImplementedError

    def send(self, task_id: str, task_type: str, payload: dict, error: str, attempts: int = 1) -> None:
        # TODO 5: implement here
        raise NotImplementedError

    def get_all(self) -> list[dict]:
        # TODO 5: implement here
        raise NotImplementedError

    def retry(self, task_id: str) -> dict | None:
        # TODO 5: implement here
        raise NotImplementedError

# ── TODO 6: Idempotency Layer ─────────────────────────────────────────────────

class IdempotencyStore:
    """
    TODO 6: Prevent duplicate processing of the same task.

    In __init__(self, db_path="idempotency.db"):
      CREATE TABLE processed (
          idempotency_key TEXT PRIMARY KEY,
          result TEXT,  -- JSON-serialized result
          created_at TEXT
      )

    Implement:
    a) is_processed(self, key: str) -> bool
    b) get_result(self, key: str) -> dict | None
       - Return cached result if key was already processed
    c) mark_processed(self, key: str, result: dict) -> None
       - Store the result

    Usage:
    idempotency_key = hashlib.sha256(f"{doc_id}:{version}".encode()).hexdigest()
    if store.is_processed(idempotency_key):
        return store.get_result(idempotency_key)
    result = await process_document(doc_id)
    store.mark_processed(idempotency_key, result)
    return result
    """
    def __init__(self, db_path: str = "idempotency.db"):
        # TODO 6: implement here
        raise NotImplementedError

    def is_processed(self, key: str) -> bool:
        # TODO 6: implement here
        raise NotImplementedError

    def get_result(self, key: str) -> dict | None:
        # TODO 6: implement here
        raise NotImplementedError

    def mark_processed(self, key: str, result: dict) -> None:
        # TODO 6: implement here
        raise NotImplementedError

async def idempotent_review(doc_id: str, version: str, chain: FallbackChain,
                             store: IdempotencyStore) -> dict:
    """
    TODO 6 (continued): Process a document review idempotently.

    Generate key = hashlib.sha256(f"{doc_id}:{version}".encode()).hexdigest()
    Check store.is_processed(key) → return cached result if yes.
    Otherwise run the review and store.mark_processed(key, result).
    """
    # TODO 6: implement here
    raise NotImplementedError

# ── TODO 7 (BONUS): Health Check ─────────────────────────────────────────────

async def check_model_health(model: str, timeout: float = 5.0) -> dict:
    """
    TODO 7: Check if a model is reachable and responding.

    Send a minimal request:
    messages = [{"role": "user", "content": "Reply: ok"}]
    Try: await asyncio.wait_for(litellm.acompletion(model, messages, max_tokens=5), timeout)
    Return: {"model": model, "healthy": True/False, "latency_ms": float, "error": str|None}
    """
    # TODO 7: implement here
    raise NotImplementedError

async def health_check_all(models: list[str] = FALLBACK_MODELS) -> None:
    """
    TODO 7 (continued): Check all models concurrently and print a health table.

    Run check_model_health for all models concurrently with asyncio.gather.
    Print:
    | Model                          | Status  | Latency |
    |--------------------------------|---------|---------|
    | openai/gpt-4o-mini             | ✅ up   | 340ms   |
    | groq/llama-3.3-70b-versatile   | ✅ up   | 120ms   |
    | ollama/llama3.2                | ❌ down | N/A     |
    """
    # TODO 7: implement here
    raise NotImplementedError

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Resilience Patterns Exercise ===\n")

    # Step 1: Circuit breaker demo
    print("1. Circuit Breaker demo...")
    cb = CircuitBreaker(name="gpt-4o-mini", failure_threshold=3, recovery_timeout=5)
    print(f"   Initial state: {cb.state.value}")
    for i in range(3):
        cb.record_failure()
        print(f"   After failure {i+1}: {cb.state.value}")
    print(f"   Can attempt: {cb.can_attempt()}")
    await asyncio.sleep(6)   # wait for recovery
    print(f"   After {6}s timeout: {cb.state.value}")

    # Step 2-3: Fallback chain
    print("\n2. Fallback chain + resilient caller...")
    chain = FallbackChain()
    resilient_call = make_resilient_caller(chain)
    try:
        text, model = await resilient_call([{"role": "user", "content": "Reply: ok"}], max_tokens=5)
        print(f"   Response from {model}: '{text.strip()}'")
    except Exception as e:
        print(f"   All models failed: {e}")
    chain.print_status()

    # Step 4: Saga
    print("\n3. Saga pattern (with compensation on failure)...")
    coordinator = SagaCoordinator()
    doc_context = {"doc_id": "DOC-999", "content": "Test vendor agreement"}

    async def review_step(ctx):
        return await run_llm_review(ctx, chain)

    steps = [
        SagaStep("reserve_slot", reserve_review_slot, release_review_slot),
        SagaStep("extract_doc", extract_document, delete_extraction),
        SagaStep("run_review", review_step, archive_failed_review),
        SagaStep("store_result", store_to_db, delete_from_db),
    ]

    try:
        result = await coordinator.execute(steps, doc_context)
        print(f"   ✅ Saga completed: {result.get('model_used', 'unknown')} model used")
    except SagaError as e:
        print(f"   ❌ Saga failed (compensated): {e}")

    # Step 5: DLQ
    print("\n4. Dead Letter Queue...")
    dlq = DeadLetterQueue()
    dlq.send("DOC-ERR-001", "compliance_review", {"doc_id": "DOC-001"}, "Timeout after 3 attempts", 3)
    dlq.print_summary()

    # Step 6: Idempotency
    print("\n5. Idempotency...")
    store = IdempotencyStore()
    r1 = await idempotent_review("DOC-100", "v1", chain, store)
    r2 = await idempotent_review("DOC-100", "v1", chain, store)  # should return cached
    print(f"   Same result returned: {r1 == r2}")

    # Step 7: Health checks
    print("\n6. Model health checks...")
    await health_check_all()

if __name__ == "__main__":
    asyncio.run(main())
