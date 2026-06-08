"""
SOLUTION — Exercise 2: Circuit Breaker + Fallback Chain + Saga Pattern
Phase 7 / Week 16

How this solution works:
  TODO 1: CircuitBreaker uses three computed states based on failure count + timer:
           CLOSED (< threshold failures) → OPEN (>= threshold, recent) → HALF_OPEN (timer expired).
  TODO 2: FallbackChain holds [primary, secondary, tertiary] models each with its own
           CircuitBreaker; tries each in order until one succeeds.
  TODO 3: make_resilient_caller wraps tenacity retry with exponential backoff around
           the fallback chain — handles transient failures before escalating.
  TODO 4: SagaCoordinator runs steps sequentially; on failure, executes compensations
           in reverse order so system returns to a consistent state.
  TODO 5: DeadLetterQueue stores permanently failed tasks in SQLite for inspection/replay.
  TODO 6: IdempotencyStore uses a hash of (function_name + inputs) to prevent duplicate
           execution — safe to retry agent runs without double side effects.
  TODO 7 (BONUS): health_check_all() tests each model with a trivial prompt and reports status.
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


# ── TODO 1 SOLUTION: Circuit Breaker ─────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 30    # seconds
    _failures: int = field(default=0, init=False)
    _opened_at: datetime | None = field(default=None, init=False)
    _total_requests: int = field(default=0, init=False)
    _total_failures: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._failures < self.failure_threshold:
            return CircuitState.CLOSED
        # _failures >= threshold: check timer
        if self._opened_at is None:
            return CircuitState.CLOSED    # shouldn't happen, but safe default
        elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
        if elapsed < self.recovery_timeout:
            return CircuitState.OPEN
        return CircuitState.HALF_OPEN

    def can_attempt(self) -> bool:
        # CLOSED: yes. HALF_OPEN: yes (let one test request through). OPEN: no.
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._total_requests += 1

    def record_failure(self) -> None:
        self._total_requests += 1
        self._total_failures += 1
        self._failures += 1
        if self._failures >= self.failure_threshold and self._opened_at is None:
            self._opened_at = datetime.now(timezone.utc)
            print(f"  ⚡ CircuitBreaker '{self.name}' OPENED after {self._failures} failures")

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._failures,
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "failure_rate": round(self._total_failures / max(1, self._total_requests), 3),
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
        }


# ── TODO 2 SOLUTION: Fallback Chain ──────────────────────────────────────────

@dataclass
class FallbackChain:
    models: list[str]   # ordered: primary first, fallbacks after

    def __post_init__(self):
        self.breakers: dict[str, CircuitBreaker] = {
            m: CircuitBreaker(name=m, failure_threshold=3, recovery_timeout=60)
            for m in self.models
        }

    async def call(self, messages: list[dict], **kwargs) -> tuple[str, str]:
        """Try each model in order. Return (response_text, model_used)."""
        last_error: Exception | None = None

        for model in self.models:
            breaker = self.breakers[model]
            if not breaker.can_attempt():
                print(f"  ⛔ Skipping '{model}' — circuit OPEN")
                continue

            try:
                resp = await litellm.acompletion(
                    model=model, messages=messages, timeout=10, **kwargs
                )
                breaker.record_success()
                return resp.choices[0].message.content.strip(), model
            except Exception as e:
                breaker.record_failure()
                last_error = e
                print(f"  ✗ '{model}' failed: {e}")

        # All models failed — raise last error
        raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}")


# ── TODO 3 SOLUTION: Resilient Caller with Retry ─────────────────────────────

def make_resilient_caller(chain: FallbackChain):
    """Wrap FallbackChain in tenacity exponential backoff retry."""
    from tenacity import (    # type: ignore
        retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
    )
    import logging
    logger = logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RuntimeError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def resilient_call(messages: list[dict], **kwargs):
        return await chain.call(messages, **kwargs)

    return resilient_call


# ── TODO 4 SOLUTION: Saga Coordinator ────────────────────────────────────────

@dataclass
class SagaStep:
    name: str
    action: Callable[..., Awaitable[Any]]
    compensation: Callable[..., Awaitable[Any]] | None = None

class SagaCoordinator:
    def __init__(self, steps: list[SagaStep]):
        self.steps = steps

    async def execute(self, context: dict) -> dict:
        """
        Execute each step in order. On failure, run compensations in reverse.
        context is a shared dict passed to each action and compensation.
        """
        completed: list[SagaStep] = []
        results: dict[str, Any] = {}

        for step in self.steps:
            print(f"  [Saga] Executing: {step.name}")
            try:
                result = await step.action(context)
                results[step.name] = result
                context[f"{step.name}_result"] = result
                completed.append(step)
                print(f"  [Saga] ✓ {step.name} succeeded")
            except Exception as e:
                print(f"  [Saga] ✗ {step.name} failed: {e}")
                print(f"  [Saga] Starting compensations (reverse order)...")

                for done_step in reversed(completed):
                    if done_step.compensation:
                        print(f"  [Saga] Compensating: {done_step.name}")
                        try:
                            await done_step.compensation(context)
                            print(f"  [Saga] ✓ Compensated: {done_step.name}")
                        except Exception as comp_err:
                            print(f"  [Saga] ✗ Compensation failed for {done_step.name}: {comp_err}")

                raise RuntimeError(f"Saga failed at step '{step.name}': {e}") from e

        print(f"  [Saga] All {len(self.steps)} steps completed successfully")
        return results


# ── TODO 5 SOLUTION: Dead Letter Queue ───────────────────────────────────────

class DeadLetterQueue:
    def __init__(self, db_path: str = "./dlq.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dlq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                task_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                error TEXT,
                attempts INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_attempted_at TEXT
            )
        """)
        self.conn.commit()

    def enqueue(self, task_id: str, task_type: str, payload: dict, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT OR REPLACE INTO dlq (task_id, task_type, payload, error, attempts, created_at, last_attempted_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
        """, (task_id, task_type, json.dumps(payload), error, now, now))
        self.conn.commit()
        print(f"  [DLQ] Task '{task_id}' added to dead letter queue: {error[:60]}")

    def get_all(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT task_id, task_type, payload, error, attempts, created_at FROM dlq ORDER BY created_at"
        ).fetchall()
        return [
            {"task_id": r[0], "task_type": r[1], "payload": json.loads(r[2]),
             "error": r[3], "attempts": r[4], "created_at": r[5]}
            for r in rows
        ]

    def delete(self, task_id: str) -> None:
        self.conn.execute("DELETE FROM dlq WHERE task_id = ?", (task_id,))
        self.conn.commit()

    def size(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM dlq").fetchone()[0]

    def close(self):
        self.conn.close()


# ── TODO 6 SOLUTION: Idempotency Store ───────────────────────────────────────

class IdempotencyStore:
    def __init__(self, db_path: str = "./idempotency.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def make_key(self, fn_name: str, **kwargs) -> str:
        payload = json.dumps({"fn": fn_name, **kwargs}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get_cached(self, key: str) -> Any | None:
        row = self.conn.execute(
            "SELECT result FROM idempotency_keys WHERE key = ?", (key,)
        ).fetchone()
        if row:
            print(f"  [Idempotency] Cache HIT for key={key[:8]}...")
            return json.loads(row[0])
        return None

    def store(self, key: str, result: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO idempotency_keys (key, result, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(result), datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()
        print(f"  [Idempotency] Stored result for key={key[:8]}...")

    def close(self):
        self.conn.close()


async def idempotent_review(
    idempotency_store: IdempotencyStore,
    chain: FallbackChain,
    doc_id: str,
    document: str,
) -> dict:
    """Compliance review that is safe to retry — same doc always returns same result."""
    key = idempotency_store.make_key("compliance_review", doc_id=doc_id)

    cached = idempotency_store.get_cached(key)
    if cached is not None:
        return cached

    print(f"  [Idempotency] Cache MISS for key={key[:8]}... executing review")
    text, model_used = await chain.call(
        messages=[
            {"role": "system", "content": "Return JSON: {\"risk_level\": \"low|medium|high|critical\", \"summary\": \"...\"}"},
            {"role": "user", "content": f"Review this document for compliance risk:\n{document}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    result = json.loads(text)
    result["model_used"] = model_used
    result["doc_id"] = doc_id

    idempotency_store.store(key, result)
    return result


# ── TODO 7 BONUS: Health Check All Models ────────────────────────────────────

async def health_check_all(chain: FallbackChain) -> dict[str, dict]:
    health: dict[str, dict] = {}

    async def check_one(model: str) -> dict:
        t0 = time.perf_counter()
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Reply with: OK"}],
                max_tokens=5,
                timeout=5,
            )
            latency = (time.perf_counter() - t0) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency, 1),
                "circuit_state": chain.breakers[model].state.value,
            }
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return {
                "status": "unhealthy",
                "error": str(e)[:80],
                "latency_ms": round(latency, 1),
                "circuit_state": chain.breakers[model].state.value,
            }

    results = await asyncio.gather(*[check_one(m) for m in chain.models])
    for model, result in zip(chain.models, results):
        health[model] = result
    return health


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Resilience Patterns — SOLUTION ===\n")

    # 1. Circuit Breaker Demo
    print("--- 1. Circuit Breaker ---")
    cb = CircuitBreaker("test_service", failure_threshold=3, recovery_timeout=5)
    print(f"  Initial state: {cb.state.value}")
    cb.record_failure(); cb.record_failure(); cb.record_failure()
    print(f"  After 3 failures: {cb.state.value}")
    print(f"  Can attempt: {cb.can_attempt()}")
    await asyncio.sleep(5.5)  # wait for recovery_timeout
    print(f"  After 5.5s: {cb.state.value} (HALF_OPEN)")
    cb.record_success()
    print(f"  After success: {cb.state.value}")
    print()

    # 2. Fallback Chain
    print("--- 2. Fallback Chain ---")
    chain = FallbackChain(models=[
        "openai/gpt-4o-mini",
        "openai/gpt-4o-mini",  # same model; in production use different providers
    ])
    caller = make_resilient_caller(chain)
    response, model = await caller(
        messages=[{"role": "user", "content": "Is GDPR an EU regulation? Answer yes/no."}]
    )
    print(f"  Response: '{response}' (via {model})\n")

    # 3. Health Check
    print("--- 3. Health Check ---")
    health = await health_check_all(chain)
    for model, status in health.items():
        print(f"  {model}: {status['status']} ({status['latency_ms']:.0f}ms)")
    print()

    # 4. Saga Pattern Demo
    print("--- 4. Saga Pattern ---")
    doc_id = "DOC-SAGA-001"
    context: dict = {"doc_id": doc_id}

    async def submit_for_review(ctx: dict) -> str:
        print(f"    → Submitted {ctx['doc_id']} for legal review")
        ctx["review_id"] = f"REVIEW-{ctx['doc_id']}"
        return ctx["review_id"]

    async def cancel_review(ctx: dict) -> None:
        if "review_id" in ctx:
            print(f"    ← Cancelled review {ctx['review_id']}")

    async def update_status_pending(ctx: dict) -> str:
        print(f"    → Status updated to PENDING")
        ctx["previous_status"] = "DRAFT"
        return "PENDING"

    async def revert_status(ctx: dict) -> None:
        print(f"    ← Reverted status to {ctx.get('previous_status', 'DRAFT')}")

    async def send_notification(ctx: dict) -> str:
        print(f"    → Notification sent to legal team")
        return "email_sent"

    async def send_cancellation(ctx: dict) -> None:
        print(f"    ← Sent cancellation notification")

    async def archive_document(ctx: dict) -> str:
        # Simulate failure on this step
        raise RuntimeError("Archive service temporarily unavailable")

    saga = SagaCoordinator(steps=[
        SagaStep("submit_review", submit_for_review, cancel_review),
        SagaStep("update_status", update_status_pending, revert_status),
        SagaStep("notify_legal", send_notification, send_cancellation),
        SagaStep("archive", archive_document, None),    # will fail → triggers compensations
    ])

    try:
        await saga.execute(context)
    except RuntimeError as e:
        print(f"  Saga failed (expected): {e}\n")

    # 5. Dead Letter Queue
    print("--- 5. Dead Letter Queue ---")
    dlq = DeadLetterQueue("./dlq_demo.db")
    dlq.enqueue("task_001", "compliance_review", {"doc_id": "DOC-X"}, "Neo4j connection timeout")
    dlq.enqueue("task_002", "email_notify", {"user": "analyst@corp.com"}, "SMTP server down")
    print(f"  DLQ size: {dlq.size()}")
    print(f"  Items: {[t['task_id'] for t in dlq.get_all()]}")
    dlq.close()
    print()

    # 6. Idempotency
    print("--- 6. Idempotency Layer ---")
    store = IdempotencyStore("./idempotency_demo.db")
    doc = "Vendor contract processing EU PII. No DPA attached. $500k value."
    print("  First call (cache miss — executes review):")
    result1 = await idempotent_review(store, chain, "DOC-IDEM-001", doc)
    print(f"  Result: {result1['risk_level']}")
    print("  Second call (cache hit — returns instantly):")
    result2 = await idempotent_review(store, chain, "DOC-IDEM-001", doc)
    print(f"  Result: {result2['risk_level']} (same, from cache)")
    assert result1["risk_level"] == result2["risk_level"]
    store.close()

if __name__ == "__main__":
    asyncio.run(main())
