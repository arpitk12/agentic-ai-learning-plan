"""
Project 34 SOLUTION — Production Resilience System
Circuit breaker + fallback chain + saga pattern + dead-letter queue + idempotency.
This is the capstone resilience agent that ties all patterns into a cohesive system.
"""
from __future__ import annotations
import os, json, asyncio, time, hashlib, sqlite3, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine
import litellm
from dotenv import load_dotenv

load_dotenv()


# ── 1. Circuit Breaker ────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # normal — requests pass through
    OPEN      = "open"       # tripped — requests fail fast
    HALF_OPEN = "half_open"  # recovery probe allowed

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3,
                 recovery_timeout: float = 30.0):
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

    def __repr__(self):
        return f"CircuitBreaker({self.name}, state={self.state.value}, failures={self._failures})"


# ── 2. Fallback Chain ─────────────────────────────────────────────────────────

class FallbackChain:
    """Tries models in priority order; uses circuit breakers to skip tripped ones."""
    def __init__(self, models: list[str]):
        self._models = models
        self._breakers = {m: CircuitBreaker(m) for m in models}

    async def call(self, messages: list[dict], **kwargs) -> tuple[str, str]:
        """Returns (reply, model_used). Raises RuntimeError if all fail."""
        for model in self._models:
            breaker = self._breakers[model]
            if not breaker.can_attempt():
                print(f"  ⏭  Skipping {model} (circuit {breaker.state.value})")
                continue
            try:
                resp = await litellm.acompletion(model=model, messages=messages, **kwargs)
                breaker.record_success()
                return resp.choices[0].message.content, model
            except Exception as e:
                print(f"  ✗ {model} failed: {e}")
                breaker.record_failure()
        raise RuntimeError("All models in fallback chain failed")

    def status(self) -> dict:
        return {m: b.state.value for m, b in self._breakers.items()}


# ── 3. Retry with Exponential Backoff ─────────────────────────────────────────

async def with_retry(
    fn: Callable[[], Coroutine],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    label: str = "operation",
) -> Any:
    """Async exponential backoff retry."""
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            print(f"  ↩ Retry {attempt+1}/{max_attempts-1} for [{label}] in {delay:.1f}s: {e}")
            await asyncio.sleep(delay)


# ── 4. Saga Coordinator ───────────────────────────────────────────────────────

@dataclass
class SagaStep:
    name: str
    action: Callable[[], Coroutine]
    compensation: Callable[[], Coroutine] | None = None

class SagaCoordinator:
    """Execute steps sequentially; on failure, reverse completed steps via compensations."""
    async def execute(self, steps: list[SagaStep]) -> list[Any]:
        completed: list[SagaStep] = []
        results: list[Any] = []
        for step in steps:
            try:
                print(f"  ▶ Saga step: {step.name}")
                result = await step.action()
                completed.append(step)
                results.append(result)
                print(f"    ✓ {step.name} done")
            except Exception as e:
                print(f"  ✗ Saga step [{step.name}] failed: {e}")
                print(f"  ↩ Rolling back {len(completed)} completed steps...")
                for done in reversed(completed):
                    if done.compensation:
                        try:
                            await done.compensation()
                            print(f"    ✓ Compensated: {done.name}")
                        except Exception as ce:
                            print(f"    ✗ Compensation failed for {done.name}: {ce}")
                raise RuntimeError(f"Saga failed at step '{step.name}': {e}") from e
        return results


# ── 5. Dead-Letter Queue ──────────────────────────────────────────────────────

class DeadLetterQueue:
    """SQLite-backed DLQ for failed requests awaiting manual review or retry."""
    def __init__(self, db_path: str = "/tmp/dlq.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS dlq (
                id TEXT PRIMARY KEY,
                payload TEXT,
                error TEXT,
                attempts INTEGER DEFAULT 1,
                ts REAL DEFAULT (unixepoch())
            )
        """)
        self._conn.commit()

    def enqueue(self, payload: dict, error: str) -> str:
        item_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO dlq (id, payload, error) VALUES (?,?,?)",
            (item_id, json.dumps(payload), error),
        )
        self._conn.commit()
        print(f"  📥 DLQ enqueued: {item_id[:8]}... error='{error[:60]}'")
        return item_id

    def get_all(self) -> list[dict]:
        rows = self._conn.execute("SELECT id, payload, error, attempts, ts FROM dlq").fetchall()
        return [{"id": r[0], "payload": json.loads(r[1]), "error": r[2],
                 "attempts": r[3], "ts": r[4]} for r in rows]

    def delete(self, item_id: str):
        self._conn.execute("DELETE FROM dlq WHERE id=?", (item_id,))
        self._conn.commit()

    def size(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM dlq").fetchone()[0]


# ── 6. Idempotency Store ──────────────────────────────────────────────────────

class IdempotencyStore:
    """Prevents duplicate processing by caching results keyed by request hash."""
    def __init__(self):
        self._cache: dict[str, Any] = {}

    def _key(self, fn_name: str, **kwargs) -> str:
        raw = json.dumps({"fn": fn_name, **kwargs}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, fn_name: str, **kwargs) -> tuple[bool, Any]:
        key = self._key(fn_name, **kwargs)
        if key in self._cache:
            return True, self._cache[key]
        return False, None

    def store(self, fn_name: str, result: Any, **kwargs):
        key = self._key(fn_name, **kwargs)
        self._cache[key] = result


# ── 7. Resilient Compliance Agent ─────────────────────────────────────────────

fallback_chain = FallbackChain([
    "openai/gpt-4o-mini",
    "openai/gpt-3.5-turbo",  # cheaper fallback
])
dlq = DeadLetterQueue()
idempotency = IdempotencyStore()
saga = SagaCoordinator()

async def resilient_compliance_agent(
    document: str,
    doc_type: str = "contract",
    request_id: str | None = None,
) -> dict:
    """
    Production-grade compliance analysis with:
    - Idempotency: same (document, doc_type) never processed twice
    - Fallback chain: GPT-4o-mini → GPT-3.5-turbo
    - Retry: up to 3 attempts with exponential backoff
    - DLQ: failed requests stored for later
    - Saga: multi-step process with rollback on failure
    """
    rid = request_id or hashlib.md5(f"{doc_type}:{document[:100]}".encode()).hexdigest()[:8]

    # Idempotency check
    cached, result = idempotency.get("compliance", document=document[:50], doc_type=doc_type)
    if cached:
        print(f"  ✅ Idempotency hit for request {rid} — returning cached result")
        return {**result, "cached": True}

    messages = [{
        "role": "system",
        "content": "You are a compliance analyzer. Return JSON only.",
    }, {
        "role": "user",
        "content": f"""Analyze this {doc_type} for compliance issues:

{document}

Return JSON: {{"risk_level": "low|medium|high|critical", "key_issues": ["..."], "required_actions": ["..."]}}""",
    }]

    # Saga pattern: extract → analyze → store
    extracted_result: dict = {}
    analysis_result: dict = {}
    stored_id = str(uuid.uuid4())

    async def step_extract():
        nonlocal extracted_result
        reply, model = await with_retry(
            lambda: fallback_chain.call(messages, response_format={"type": "json_object"}, temperature=0.0),
            label="extract",
        )
        extracted_result = {"raw": json.loads(reply), "model": model, "request_id": rid}
        return extracted_result

    async def step_analyze():
        nonlocal analysis_result
        analysis_result = {
            **extracted_result["raw"],
            "model_used": extracted_result["model"],
            "request_id": rid,
        }
        return analysis_result

    async def step_store():
        # Simulate storing to database
        idempotency.store("compliance", analysis_result, document=document[:50], doc_type=doc_type)
        return stored_id

    async def compensate_store():
        idempotency._cache.pop(
            idempotency._key("compliance", document=document[:50], doc_type=doc_type), None
        )
        print(f"    Rolled back stored result {stored_id[:8]}...")

    try:
        await saga.execute([
            SagaStep("extract_document", step_extract),
            SagaStep("analyze_compliance", step_analyze),
            SagaStep("store_result", step_store, compensate_store),
        ])
        return analysis_result
    except Exception as e:
        payload = {"document": document[:200], "doc_type": doc_type, "request_id": rid}
        dlq.enqueue(payload, str(e))
        return {"error": str(e), "queued_for_retry": True, "dlq_size": dlq.size()}


# ── 8. Health Check ───────────────────────────────────────────────────────────

async def health_check_all() -> dict:
    """Check health of all models in the fallback chain."""
    results = {}
    async def probe(model: str):
        start = time.time()
        try:
            resp = await litellm.acompletion(
                model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=5
            )
            latency = time.time() - start
            state = fallback_chain._breakers[model].state.value
            results[model] = {"status": "healthy", "latency_ms": round(latency*1000), "circuit": state}
        except Exception as e:
            results[model] = {"status": "unhealthy", "error": str(e),
                              "circuit": fallback_chain._breakers[model].state.value}

    await asyncio.gather(*[probe(m) for m in fallback_chain._models])
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Project 34: Production Resilience SOLUTION ===\n")

    # Demo circuit breaker
    print("1. Circuit Breaker demo:")
    cb = CircuitBreaker("demo", failure_threshold=3, recovery_timeout=30)
    print(f"  Initial state: {cb.state.value}")
    cb.record_failure(); cb.record_failure(); cb.record_failure()
    print(f"  After 3 failures: {cb.state.value}")
    cb.record_success()
    print(f"  After success: {cb.state.value}")

    # Demo fallback chain status
    print("\n2. Fallback chain status:")
    for model, state in fallback_chain.status().items():
        print(f"  {model}: {state}")

    # Demo DLQ
    print("\n3. Dead-Letter Queue demo:")
    print(f"  DLQ size before: {dlq.size()}")
    did = dlq.enqueue({"document": "test contract"}, "timeout after 3 attempts")
    print(f"  DLQ size after: {dlq.size()}")
    dlq.delete(did)
    print(f"  DLQ size after delete: {dlq.size()}")

    # Demo idempotency
    print("\n4. Idempotency demo:")
    idempotency.store("test_fn", {"result": "ok"}, doc="abc", type="x")
    hit, cached = idempotency.get("test_fn", doc="abc", type="x")
    print(f"  Cache hit: {hit}, result: {cached}")

    # Demo resilient agent
    print("\n5. Resilient compliance agent:")
    result = await resilient_compliance_agent(
        document="Software license agreement between TechCorp and CloudProvider. "
                 "No limitation of liability clause. GDPR DPA missing. EU data transfer without SCCs. "
                 "Contract value: $500K/year.",
        doc_type="contract",
        request_id="req-001",
    )
    if "error" not in result:
        print(f"  Risk: {result.get('risk_level')}")
        print(f"  Issues: {result.get('key_issues', [])[:2]}")
        print(f"  Model: {result.get('model_used')}")

    # Test idempotency on second call
    print("\n6. Idempotency — same request:")
    result2 = await resilient_compliance_agent(
        document="Software license agreement between TechCorp and CloudProvider. "
                 "No limitation of liability clause. GDPR DPA missing. EU data transfer without SCCs. "
                 "Contract value: $500K/year.",
        doc_type="contract",
    )
    print(f"  Cached: {result2.get('cached', False)}")

    # Health check
    print("\n7. Model health check:")
    health = await health_check_all()
    for model, status in health.items():
        icon = "✅" if status["status"] == "healthy" else "❌"
        print(f"  {icon} {model}: {status.get('latency_ms', '?')}ms | circuit={status['circuit']}")

if __name__ == "__main__":
    asyncio.run(main())
