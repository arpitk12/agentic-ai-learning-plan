# Project 40 — Scale & Load Testing

> **Stack**: Locust · FastAPI · asyncio · Redis · Python 3.11+  
> **Theme**: System Design — Chapter 10 of `guide/13_system_design.md`  
> **Companion guide**: [`guide/13_system_design.md §10`](../../guide/13_system_design.md)  
> **Prerequisite**: Complete project39_async_platform first (this project load-tests it)

---

## What You'll Build

A **load testing + capacity planning suite** that finds the breaking points of your agent API, identifies the bottlenecks, fixes them, and produces a capacity plan for production.

```
Phase 1: Baseline load test
  Locust → 10/50/100/200 concurrent users → Agent API
  Measure: p50, p95, p99 latency | RPS | error rate | cost/request

Phase 2: Bottleneck analysis
  Identify: DB contention | LLM rate limits | Redis pub/sub lag | worker starvation

Phase 3: Fix and re-test
  Implement: connection pooling | response caching | rate limiter |
             worker auto-scale | LLM request queue

Phase 4: Capacity plan
  Plot: concurrent users vs RPS (with/without fixes)
  Output: "At 95th percentile <5s: system supports N concurrent users
           with M Celery workers and K Redis connections"
```

---

## Why This Project Matters

Every system breaks at scale. The only question is whether you find the breaking point in a load test or in production at 3am. This project teaches:
- **How to measure**: Locust + structured metrics (not just "it felt slow")
- **How to diagnose**: which layer is the bottleneck (CPU? network? DB? LLM rate limit?)
- **How to fix**: the 5 most common agent API bottlenecks
- **How to plan**: translate test results into capacity requirements for your SLA

---

## System Design Concepts Covered

| Concept | Where |
|---|---|
| Load testing with Locust | `locustfile.py` |
| Response caching (identical queries) | `CacheMiddleware` |
| DB connection pooling | `pool_size`, `max_overflow` in SQLModel engine |
| LLM request queue (prevent thundering herd) | `LLMRequestQueue` |
| Rate limiting per user (token bucket) | `RateLimiter` |
| Horizontal worker scaling | `celery_hpa.py` (metrics + scale decision) |
| Capacity planning formula | `capacity_plan.py` |
| SLA definition (p95 < 5s) | `sla.py` |
| Before/after benchmark comparison | `compare_results.py` |

---

## Milestones

### Milestone 1 — Baseline Locust Test
Write `locustfile.py`. User class with tasks: `submit_run` (POST /agent/run), `poll_status` (GET /status), `get_result` (GET /result). Run at 10, 50, 100 concurrent users. Record: p50, p95 latency; RPS; error rate; total cost.

### Milestone 2 — Metrics Endpoint
Add `GET /metrics` to the FastAPI app (from project39). Returns JSON:
```json
{
  "active_runs": 12,
  "queue_depth": 47,
  "worker_count": 4,
  "avg_cost_usd": 0.00183,
  "p95_latency_ms": 4200,
  "error_rate_pct": 0.2
}
```

### Milestone 3 — Response Cache
Implement `ResponseCache`. For identical queries (SHA-256 hash), return cached result within TTL=300s. Measure cache hit rate under load. Show cost reduction from caching.

### Milestone 4 — DB Connection Pooling
Configure SQLModel engine with `pool_size=20, max_overflow=40`. Compare DB query latency under 100 concurrent users with default vs pooled connections.

### Milestone 5 — LLM Request Queue (Thundering Herd Fix)
Implement `LLMRequestQueue` — an asyncio Semaphore that limits concurrent LLM calls to `max_concurrent=10`. Measure: without queue at 100 users, LLM rate limit errors. With queue: rate limit errors drop to 0, p95 latency improves.

### Milestone 6 — Per-User Rate Limiter
Implement Redis token bucket rate limiter: 10 requests/minute (free tier). Test: single user sends 20 requests/min → 10 succeed, 10 get 429. Verify rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After`.

### Milestone 7 — Worker Auto-Scaling Simulator
Implement `WorkerScaler` that reads queue depth from Redis every 10s. Scale-up rule: queue_depth > 5 × worker_count → add workers. Scale-down rule: queue_depth < 2 × worker_count for 60s → remove workers. Simulate with mock queue depth data. Produce a "workers over time" chart.

### Milestone 8 — Capacity Plan Report
Implement `CapacityPlanner.generate_report(test_results)`. For each load level tested, record: concurrent_users, rps, p95_latency_ms, error_rate, worker_count. Find the maximum concurrent users that keeps p95 < 5000ms and error_rate < 1%. Output a table and recommendation.

### Milestone 9 — Before/After Comparison
Run full load test (100 users) before and after all fixes. Produce side-by-side comparison:

| Metric | Before | After | Improvement |
|---|---|---|---|
| p95 latency | 12.4s | 3.8s | 69% faster |
| Error rate | 8.2% | 0.1% | 99% fewer errors |
| RPS | 2.3 | 9.1 | 4× throughput |
| Max concurrent users | 20 | 85 | 4.25× capacity |

---

## Setup

```bash
pip install locust fastapi uvicorn redis sqlmodel litellm pydantic matplotlib

# Requires project39 running:
cd ../project39_async_platform
uvicorn starter.app:app --port 8000 &
celery -A starter.tasks worker --concurrency=4 &

# Run Locust
cd ../project40_scale_loadtest
locust -f locustfile.py --host=http://localhost:8000 --headless \
       --users=100 --spawn-rate=10 --run-time=2m
```

---

## Expected Output (Milestone 8 — Capacity Plan)

```
═══════════════════════════════════════════════════════════════
 CAPACITY PLANNING REPORT
═══════════════════════════════════════════════════════════════

 Test results (after fixes, 4 Celery workers):

 Users │  RPS   │ p50 ms │ p95 ms │ Error % │ SLA ✅/❌
───────┼────────┼────────┼────────┼─────────┼──────────
   10  │  4.2   │  1,840 │  3,100 │  0.0%   │ ✅
   25  │  8.1   │  2,100 │  4,200 │  0.0%   │ ✅
   50  │  9.4   │  3,800 │  4,900 │  0.1%   │ ✅
   75  │  9.8   │  4,100 │  5,400 │  0.4%   │ ❌ (p95 > 5s)
  100  │  9.1   │  5,200 │  8,100 │  2.1%   │ ❌

 SLA: p95 latency < 5,000ms, error rate < 1%

 ✅ Max supported: 50 concurrent users with 4 workers
 🔧 To support 100 users: scale to 8+ Celery workers
 📈 Scaling formula: workers_needed = ceil(concurrent_users / 12)

 Cost at max capacity:
   50 users × 5 RPS × $0.0018/req = $0.45/min = $648/month
```

---

## Stretch Goals

- [ ] Add Grafana dashboard fed by Prometheus metrics (agent_requests_total, agent_latency_seconds, agent_cost_usd_total)
- [ ] Simulate LLM provider failure mid-test: verify circuit breaker activates and fallback model handles traffic
- [ ] Implement distributed load test across 2 Locust workers (master + worker mode)
- [ ] Add chaos testing: randomly kill a Celery worker mid-run; verify jobs are re-queued and completed
- [ ] Generate a PDF capacity report with matplotlib charts (RPS vs users, latency distribution)
