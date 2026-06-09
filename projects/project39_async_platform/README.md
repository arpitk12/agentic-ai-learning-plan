# Project 39 — Full Async Agent Platform

> **Stack**: FastAPI · Celery · Redis · PostgreSQL (SQLite for dev) · asyncio · SSE  
> **Theme**: System Design — Chapter 9 of `guide/13_system_design.md`  
> **Companion guide**: [`guide/13_system_design.md §9`](../../guide/13_system_design.md)

---

## What You'll Build

The **complete async agent execution platform** — from HTTP request to streaming result — that every production agent system needs but rarely implements correctly.

```
Client
  │
  ├─ POST /agent/run          → 202 Accepted + {run_id}
  │      └─ validates input
  │      └─ creates DB record (status: pending)
  │      └─ enqueues Celery task
  │
  ├─ GET  /agent/run/{id}/status    → {status, steps, cost_so_far, eta_s}
  │
  ├─ GET  /agent/run/{id}/stream    → SSE stream of step events
  │      └─ "step:1:thinking: I need to search for..."
  │      └─ "step:2:tool_call: web_search(GDPR Article 28)"
  │      └─ "step:3:tool_result: ..."
  │      └─ "done: <final answer>"
  │
  └─ GET  /agent/run/{id}/result    → {answer, cost, steps, latency_ms}


Celery Worker (runs the actual agent)
  ├─ Picks task from Redis queue
  ├─ Runs agent loop (max 10 steps, 120s timeout)
  ├─ Publishes each step to Redis pub/sub (for SSE)
  ├─ Checkpoints state every step (for crash recovery)
  ├─ Stores result in DB
  └─ Handles retries (exponential backoff, max 3)

Idempotency (prevents duplicate runs on client retries):
  POST /agent/run
    Header: Idempotency-Key: <uuid>
    → same key + same payload → return cached response (24h TTL)
```

---

## Why This Project Matters

Projects 1–6 all use synchronous `agent_loop()`. This works for CLIs and demos. In production:
- HTTP requests timeout after 30–60s (agents routinely take longer)
- Users need progress visibility (not a blank screen for 90s)
- Workers need to be horizontally scalable (one API server can't run 50 concurrent agents)
- Crashes happen; agents must resume, not restart

This project builds the infrastructure that makes all other agents production-ready.

---

## System Design Concepts Covered

| Concept | Where in code |
|---|---|
| Async HTTP with 202 + polling | `POST /agent/run` + `GET /status` |
| SSE (Server-Sent Events) streaming | `GET /stream` — `EventSourceResponse` |
| Task queue (Celery + Redis) | `tasks.py` — `@celery_app.task` |
| Agent checkpointing (crash recovery) | `AgentCheckpoint` → Redis |
| Idempotency keys (safe retries) | `IdempotencyMiddleware` |
| Per-step pub/sub (Redis) | `StepPublisher` + `StepSubscriber` |
| Database schema for agent runs | `AgentRun` model + SQLite |
| Cost tracking per run | `CostTracker` → DB update |
| Graceful shutdown | `lifespan` context manager |

---

## Milestones

### Milestone 1 — Database Schema
Implement `AgentRun` table using SQLModel (SQLite for dev). Fields: id (UUID), status (pending/running/done/failed/cancelled), query, model, input_tokens, output_tokens, cost_usd, steps, error, started_at, ended_at. Add `AgentStep` table for step-level trace.

### Milestone 2 — Celery Setup
Configure Celery with Redis broker and Redis result backend. Implement `run_agent_task` Celery task with: `soft_time_limit=120`, `time_limit=150`, `max_retries=3`, `acks_late=True`.

### Milestone 3 — Step Publisher/Subscriber
Implement `StepPublisher.publish(run_id, event_type, content)` — writes to Redis pub/sub channel `agent:{run_id}:steps`. Implement `StepSubscriber.listen(run_id)` — async generator that yields events until "done" or "error".

### Milestone 4 — Agent Checkpointing
Implement `AgentCheckpoint`. Methods:
- `save(run_id, step, messages, cost)` — serialize to Redis with TTL=3600
- `load(run_id)` — deserialize from Redis, return None if not found
- `clear(run_id)` — delete checkpoint on completion

### Milestone 5 — Agent Loop (Celery Worker)
Implement the agent loop inside the Celery task:
1. Load checkpoint (resume if exists)
2. Loop: LLM call → publish step event → execute tools → update DB → checkpoint
3. On completion: clear checkpoint, update DB (status=done, cost, tokens), publish done event
4. On error: update DB (status=failed, error), publish error event

### Milestone 6 — FastAPI Routes
Implement four routes:
- `POST /agent/run` — validate input, check idempotency, create DB record, enqueue task, return 202
- `GET /agent/run/{run_id}/status` — read DB record, return status + progress
- `GET /agent/run/{run_id}/stream` — `EventSourceResponse` that reads from `StepSubscriber`
- `GET /agent/run/{run_id}/result` — return full result (only when done)

### Milestone 7 — Idempotency Middleware
Implement `IdempotencyMiddleware`. On POST with `Idempotency-Key` header:
1. Hash (key + path + body) → Redis lookup
2. If found: return cached response immediately
3. If not found: process normally, cache response with 24h TTL

### Milestone 8 — Integration Test
Write a test script (no pytest) that:
1. POSTs a run → gets `run_id`
2. Polls `/status` until `running`
3. Opens SSE stream → prints each event as it arrives
4. Waits for `done` event
5. Fetches `/result` → prints answer + cost + steps
6. POSTs same request again with same Idempotency-Key → verifies same `run_id` returned (idempotency)

---

## Setup

```bash
# Dependencies
pip install fastapi uvicorn[standard] celery redis sqlmodel asyncio-sse python-dotenv litellm pydantic

# Start Redis (required)
docker run -d -p 6379:6379 redis:7

# Terminal 1: Start Celery worker
celery -A starter.tasks worker --loglevel=info --concurrency=4

# Terminal 2: Start FastAPI server
uvicorn starter.app:app --reload --port 8000

# Terminal 3: Run integration test
python starter/test_client.py
```

---

## Expected Output (test_client.py)

```
POST /agent/run → 202
  run_id: 9a3f1b2c-...
  status: pending

Polling status...
  [0.5s] status: pending
  [1.0s] status: running (step 1)

Streaming events:
  [step:1] thinking: I need to search for GDPR Article 28 requirements...
  [step:2] tool_call: web_search("GDPR Article 28 Data Processing Agreement")
  [step:3] tool_result: GDPR Article 28 requires a written DPA between...
  [step:4] thinking: I have enough information to answer...
  [done] GDPR Article 28 requires a Data Processing Agreement (DPA)...

Result:
  answer: GDPR Article 28 requires a Data Processing Agreement...
  cost:   $0.00183
  steps:  4
  latency_ms: 8421

Idempotency test:
  POST with same Idempotency-Key → run_id: 9a3f1b2c-... (same ✅ — cached)
```

---

## Stretch Goals

- [ ] Add WebSocket endpoint (upgrade from SSE for bidirectional)
- [ ] Implement `POST /agent/run/{run_id}/cancel` → revoke Celery task, update DB to cancelled
- [ ] Add priority queue (premium users get a higher-priority Celery queue)
- [ ] Add rate limiting per user (token bucket in Redis, 10 runs/min free tier)
- [ ] Replace SQLite with PostgreSQL; add `agent_steps` table; build `/admin/runs` dashboard
- [ ] Add OpenTelemetry trace ID that propagates from HTTP request → Celery task → each LLM call
