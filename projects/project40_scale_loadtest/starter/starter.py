"""
Project 40 — Scale & Load Testing (starter)
============================================
Load test the agent API from project39, identify bottlenecks,
implement fixes, and produce a capacity plan.

Companion: guide/13_system_design.md §10 — Scalability Patterns

This file contains:
  - locustfile.py contents (run with locust)
  - ResponseCache
  - LLMRequestQueue
  - RateLimiter
  - WorkerScaler
  - CapacityPlanner
  - compare_results (before/after)

Fill in every # TODO block. Do NOT look at solution/ until you've tried.
"""
from __future__ import annotations
import asyncio, hashlib, json, math, random, time
from dataclasses import dataclass, field
from typing import Any
import redis.asyncio as aioredis
import redis as sync_redis
from dotenv import load_dotenv

load_dotenv()
REDIS_URL = "redis://localhost:6379/0"
redis_sync = sync_redis.from_url(REDIS_URL, decode_responses=True)


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Locust Load Test (save as locustfile.py and run with locust)
# ══════════════════════════════════════════════════════════════════════════════

LOCUSTFILE_CONTENT = '''
"""
locustfile.py — Agent API load test
Run: locust -f locustfile.py --host=http://localhost:8000 --headless
         --users=100 --spawn-rate=10 --run-time=2m
"""
import time, uuid
from locust import HttpUser, task, between, events

QUERIES = [
    "What are the requirements of GDPR Article 28?",
    "Does GDPR apply to US companies?",
    "What is the right to erasure under GDPR?",
    "What fines can be imposed under GDPR?",
    "What is a Data Protection Officer?",
]

class AgentUser(HttpUser):
    wait_time = between(1, 3)   # random wait between tasks (realistic pacing)
    run_id: str | None = None

    @task(3)  # 3x more likely than other tasks (submit is the main action)
    def submit_and_poll(self):
        """Submit a run, poll until running, then stream or get result."""
        # TODO 1: POST /agent/run with a random QUERIES item
        # Record submit latency
        # Extract run_id from 202 response
        # Poll GET /agent/run/{run_id}/status until status != "pending" (max 5 polls, 1s apart)
        # GET /agent/run/{run_id}/result when done
        pass  # replace with implementation

    @task(1)
    def status_check(self):
        """Poll status of a known run_id (uses one saved from previous submit)."""
        # TODO 2: If self.run_id is set, GET /agent/run/{self.run_id}/status
        # Silently skip if no run_id
        pass

    @task(1)
    def metrics_check(self):
        """Check platform metrics endpoint."""
        # TODO 3: GET /metrics — just check it returns 200
        pass


@events.test_start.add_listener
def on_start(environment, **kwargs):
    print("Load test starting...")

@events.test_stop.add_listener
def on_stop(environment, **kwargs):
    stats = environment.stats
    print(f"\\n=== Load Test Results ===")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failures:       {stats.total.num_failures}")
    print(f"p50 latency:    {stats.total.get_response_time_percentile(0.50):.0f}ms")
    print(f"p95 latency:    {stats.total.get_response_time_percentile(0.95):.0f}ms")
    print(f"p99 latency:    {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print(f"RPS:            {stats.total.total_rps:.1f}")
'''


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Response Cache
# ══════════════════════════════════════════════════════════════════════════════

class ResponseCache:
    """
    Cache agent results for identical queries (SHA-256 hash of query).
    Reduces LLM calls and cost for repeated questions.
    """
    TTL = 300  # 5 minutes

    def _cache_key(self, query: str) -> str:
        return f"cache:query:{hashlib.sha256(query.encode()).hexdigest()}"

    async def get(self, query: str) -> dict | None:
        """Return cached result or None."""
        # TODO 4: r = await aioredis.from_url(REDIS_URL); GET cache_key; json.loads if found
        raise NotImplementedError

    async def set(self, query: str, result: dict):
        """Cache result with TTL."""
        # TODO 5: SETEX cache_key TTL json.dumps(result)
        raise NotImplementedError

    async def hit_rate(self) -> float:
        """Return cache hit rate from tracked counters."""
        # TODO 6: GET cache:hits and cache:total; return hits/total (0.0 if no data)
        raise NotImplementedError

    async def track_hit(self):
        # TODO 7: INCR cache:hits and cache:total
        raise NotImplementedError

    async def track_miss(self):
        # TODO 8: INCR cache:total (not cache:hits)
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# PART C — LLM Request Queue (Thundering Herd Prevention)
# ══════════════════════════════════════════════════════════════════════════════

class LLMRequestQueue:
    """
    Limits concurrent LLM calls using asyncio.Semaphore.
    Without this, 100 concurrent users each fire an LLM call → rate limit errors.
    With this, at most max_concurrent calls run simultaneously; others wait.
    """
    def __init__(self, max_concurrent: int = 10):
        # TODO 9: self._sem = asyncio.Semaphore(max_concurrent)
        raise NotImplementedError

    async def __aenter__(self):
        # TODO 10: await self._sem.acquire()
        raise NotImplementedError

    async def __aexit__(self, *args):
        # TODO 11: self._sem.release()
        raise NotImplementedError


# Usage in agent loop (replace direct litellm call):
# async with llm_queue:
#     resp = await litellm.acompletion(...)
llm_queue = LLMRequestQueue(max_concurrent=10)


# ══════════════════════════════════════════════════════════════════════════════
# PART D — Per-User Rate Limiter (Token Bucket in Redis)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    limit: int
    retry_after: int   # seconds until next allowed request


class RateLimiter:
    """
    Token bucket rate limiter using Redis.
    Default: 10 requests/minute per user.
    """
    def __init__(self, requests_per_minute: int = 10):
        self.rpm = requests_per_minute
        self.window = 60  # seconds

    async def check(self, user_id: str) -> RateLimitResult:
        """
        Returns RateLimitResult. Uses Redis INCR + EXPIRE.
        Key: ratelimit:{user_id}:{current_minute}
        """
        # TODO 12:
        # current_minute = int(time.time() // 60)
        # key = f"ratelimit:{user_id}:{current_minute}"
        # r = await aioredis.from_url(REDIS_URL)
        # pipe = r.pipeline()
        # pipe.incr(key)
        # pipe.expire(key, self.window + 5)
        # count, _ = await pipe.execute()
        # allowed = count <= self.rpm
        # remaining = max(0, self.rpm - count)
        # retry_after = self.window - (int(time.time()) % self.window) if not allowed else 0
        # return RateLimitResult(allowed, remaining, self.rpm, retry_after)
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# PART E — Worker Auto-Scaling Simulator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScalingEvent:
    timestamp: float
    queue_depth: int
    worker_count: int
    action: str   # "scale_up" | "scale_down" | "no_change"


class WorkerScaler:
    """
    Simulates Kubernetes HPA logic for Celery workers.
    Scale up when queue_depth > 5 × worker_count.
    Scale down when queue_depth < 2 × worker_count for 60s straight.
    """
    def __init__(self, min_workers: int = 2, max_workers: int = 50):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self._workers = min_workers
        self._low_load_since: float | None = None
        self._history: list[ScalingEvent] = []

    def tick(self, queue_depth: int, timestamp: float | None = None) -> ScalingEvent:
        """
        Call every 10 seconds with current queue depth.
        Returns a ScalingEvent with the action taken.
        """
        ts = timestamp or time.time()
        action = "no_change"

        # TODO 13:
        # Scale UP rule: if queue_depth > 5 * self._workers and self._workers < self.max_workers:
        #   self._workers = min(self._workers * 2, self.max_workers)
        #   action = "scale_up"
        #   self._low_load_since = None
        #
        # Scale DOWN rule: if queue_depth < 2 * self._workers:
        #   if self._low_load_since is None: self._low_load_since = ts
        #   elif ts - self._low_load_since > 60 and self._workers > self.min_workers:
        #     self._workers = max(self._workers // 2, self.min_workers)
        #     action = "scale_down"
        #     self._low_load_since = None
        # else: self._low_load_since = None

        event = ScalingEvent(ts, queue_depth, self._workers, action)
        self._history.append(event)
        return event

    def simulate(self, queue_depths: list[int], interval_s: float = 10.0) -> list[ScalingEvent]:
        """Run tick() for each queue depth value with simulated timestamps."""
        # TODO 14: iterate queue_depths, call tick with simulated timestamps
        raise NotImplementedError

    def render(self):
        """Print a summary of scaling events."""
        # TODO 15: print events where action != "no_change"; print final worker count
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# PART F — Capacity Planner
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoadTestResult:
    concurrent_users: int
    rps: float
    p50_ms: float
    p95_ms: float
    error_rate_pct: float
    worker_count: int
    phase: str = "after"   # "before" or "after" (before/after fixes)


class CapacityPlanner:
    P95_SLA_MS     = 5_000   # 5 seconds
    ERROR_RATE_SLA = 1.0     # 1%

    def max_supported_users(self, results: list[LoadTestResult]) -> int:
        """Return highest concurrent_users that passes both SLA thresholds."""
        # TODO 16: Filter results that pass both SLAs; return max concurrent_users (0 if none)
        raise NotImplementedError

    def workers_needed(self, target_users: int) -> int:
        """Capacity formula: ceil(target_users / 12)"""
        # TODO 17: return math.ceil(target_users / 12)
        raise NotImplementedError

    def monthly_cost(self, concurrent_users: int, rps: float, cost_per_request: float = 0.0018) -> float:
        """Estimate monthly LLM API cost."""
        # TODO 18: concurrent_users × rps × cost_per_request × 60 × 60 × 24 × 30
        raise NotImplementedError

    def generate_report(self, results: list[LoadTestResult]):
        """Print formatted capacity planning report."""
        # TODO 19: Print table of results; mark SLA pass/fail; print max users, scaling formula, cost
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# PART G — Before/After Comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_results(before: list[LoadTestResult], after: list[LoadTestResult]):
    """
    Print a before/after comparison table.
    Match results by concurrent_users.
    Show % improvement for p95, error rate, RPS.
    """
    # TODO 20: For each user count in both, print improvement row
    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Demo (mock data — replace with real Locust results)
# ══════════════════════════════════════════════════════════════════════════════

MOCK_BEFORE = [
    LoadTestResult(10,   2.1,  4_200,  9_800, 0.0, 4, "before"),
    LoadTestResult(25,   2.3,  6_100, 14_200, 1.2, 4, "before"),
    LoadTestResult(50,   2.1,  8_900, 19_400, 6.1, 4, "before"),
    LoadTestResult(100,  1.8, 12_400, 28_100, 18.3, 4, "before"),
]

MOCK_AFTER = [
    LoadTestResult(10,   4.2,  1_840,  3_100, 0.0, 4, "after"),
    LoadTestResult(25,   8.1,  2_100,  4_200, 0.0, 4, "after"),
    LoadTestResult(50,   9.4,  3_800,  4_900, 0.1, 4, "after"),
    LoadTestResult(75,   9.8,  4_100,  5_400, 0.4, 4, "after"),
    LoadTestResult(100,  9.1,  5_200,  8_100, 2.1, 8, "after"),
]

# Simulate realistic queue depth for auto-scaling test
MOCK_QUEUE = (
    [2]*6 +       # quiet period (2 min)
    [15]*12 +     # traffic spike (2 min) — should scale up
    [80]*18 +     # heavy load (3 min) — further scale up
    [5]*12 +      # load drops (2 min) — should scale down after 60s
    [1]*6         # quiet again
)


async def main():
    print("=" * 60)
    print("Project 40 — Scale & Load Testing")
    print("=" * 60)

    # Part B: Cache demo
    print("\n── Response Cache ──")
    cache = ResponseCache()
    test_query = "What are GDPR Article 28 requirements?"
    await cache.set(test_query, {"answer": "DPA required between controller and processor", "cost_usd": 0.0018})
    cached = await cache.get(test_query)
    print(f"  Cache hit: {cached is not None} ({'✅' if cached else '❌'})")
    await cache.track_hit()
    await cache.track_miss()
    hit_rate = await cache.hit_rate()
    print(f"  Hit rate: {hit_rate:.0%}")

    # Part D: Rate limiter demo
    print("\n── Rate Limiter ──")
    limiter = RateLimiter(requests_per_minute=10)
    for i in range(12):
        result = await limiter.check("user-123")
        status = "✅ allowed" if result.allowed else f"❌ 429 (retry in {result.retry_after}s)"
        print(f"  Request {i+1:2d}: {status} | remaining={result.remaining}")

    # Part E: Auto-scaling simulator
    print("\n── Worker Auto-Scaler ──")
    scaler = WorkerScaler(min_workers=2, max_workers=32)
    events = scaler.simulate(MOCK_QUEUE)
    scaler.render()

    # Part F: Capacity plan
    print("\n── Capacity Plan ──")
    planner = CapacityPlanner()
    planner.generate_report(MOCK_AFTER)

    # Part G: Before/After
    print("\n── Before vs After Fixes ──")
    compare_results(MOCK_BEFORE, MOCK_AFTER)

    # Print locustfile instructions
    print("\n── Locust Usage ──")
    print("  Save this as locustfile.py (see LOCUSTFILE_CONTENT at top of this file)")
    print("  Start project39 first, then:")
    print("  locust -f locustfile.py --host=http://localhost:8000 --headless")
    print("         --users=100 --spawn-rate=10 --run-time=2m")


if __name__ == "__main__":
    asyncio.run(main())
