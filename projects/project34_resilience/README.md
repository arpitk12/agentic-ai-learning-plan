# Project 34 — Failure Resilience (Circuit Breaker + Saga + DLQ)

> **Stack**: Tenacity · LiteLLM · Celery · Redis · asyncio  
> **Phase 7 — Advanced Production** | Priority: P2 🟡

---

## What You'll Build

A production resilience layer that keeps your agent running even when LLMs are down, tools fail mid-workflow, or tasks need to be retried safely.

```
Incoming request
    │
    ▼ Circuit Breaker       ← stops hammering failing APIs
    │  CLOSED → OPEN after 5 failures
    │  OPEN → HALF-OPEN after 30s
    ▼ Fallback Chain        ← escalating to cheaper/local models
    │  GPT-4o → Claude → Groq (free) → Ollama (local)
    ▼ Retry (exponential)   ← tenacity: 1s, 2s, 4s, 8s backoff
    │
    ▼ Saga (multi-step)     ← rollback if any step fails
    │  step1 → step2 → step3
    │  fail at step3 → compensate step2 → compensate step1
    ▼
    ▼ Dead Letter Queue     ← permanently failed tasks for manual review
    ▼
Result (or DLQ entry)
```

---

## Milestones

### Milestone 1 — Circuit Breaker
Implement the state machine: CLOSED / OPEN / HALF_OPEN. Test: trigger with 5 rapid failures, verify OPEN state, wait for recovery timeout, verify HALF_OPEN, verify recovery to CLOSED.

### Milestone 2 — Model Fallback Chain
Four-model chain with per-model circuit breakers: GPT-4o-mini → Claude Haiku → Groq Llama → Ollama. Simulate failures by patching litellm. Verify correct fallback order.

### Milestone 3 — Retry with Tenacity
Wrap fallback chain with `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))`. Add `asyncio.wait_for(timeout=30)`. Test retry logging.

### Milestone 4 — Saga Pattern
Build `SagaCoordinator.execute(steps)`. Test: 5-step workflow where step 3 fails → verify steps 1+2 are compensated in reverse order. Test: step 5 fails → all 4 preceding steps compensated.

### Milestone 5 — Dead Letter Queue
Celery + Redis DLQ for tasks that fail after all retries. Schema: task_id, task_type, payload, error, attempt_count, timestamps. Admin API to inspect + retry DLQ entries.

### Milestone 6 — Idempotency Store
SQLite-backed idempotency: hash task inputs → check if already processed → return cached result. Test: submit same task twice in parallel → verify processed only once, second call returns cached.

### Milestone 7 — Chaos Testing
Build a chaos test suite: randomly inject failures at different layers (10%, 30%, 50% failure rates). Measure: error rate to user (should be near 0%), fallback usage rate, average latency under chaos.

---

## Setup

```bash
pip install tenacity litellm celery redis asyncio python-dotenv pydantic
docker run -p 6379:6379 redis:7
celery -A solution.tasks worker --loglevel=info
```

---

## Expected Output

```
=== Resilience Test Suite ===

Circuit Breaker:
  Failures: [1, 2, 3, 4, 5] → state: OPEN (opened_at: 14:23:01)
  After 30s: state: HALF_OPEN
  Test request succeeds → state: CLOSED ✅

Fallback Chain (simulated 70% primary failure rate):
  Request 1: gpt-4o-mini ✅ (CLOSED)
  Request 2: gpt-4o-mini ❌ → claude-haiku ✅ (fallback used)
  Request 3: gpt-4o-mini ❌ → claude-haiku ❌ → groq ✅ (2x fallback)

Saga (failure at step 3/5):
  ✓ step1: reserve_slot
  ✓ step2: extract_doc
  ✗ step3: llm_review FAILED
  ← compensate step2: delete_extraction
  ← compensate step1: release_slot
  SagaError raised. 0 side effects remaining ✅

Chaos test (50% failure injection, 100 requests):
  Error rate to user: 2% (2/100 fell through all fallbacks)
  Fallback usage: 48% (groq used 31%, ollama used 17%)
  Avg latency: 1.2s (vs 0.8s without chaos — 50% overhead)
```
