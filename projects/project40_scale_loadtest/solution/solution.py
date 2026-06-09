"""
Project 40 — Scale & Load Testing (SOLUTION)
=============================================
Full implementation of all TODOs from starter.py.
Run: python solution.py
"""
from __future__ import annotations
import asyncio, hashlib, json, math, random, time
from dataclasses import dataclass, field
from typing import Any
import redis.asyncio as aioredis
import redis as sync_redis
from dotenv import load_dotenv

load_dotenv()
REDIS_URL  = "redis://localhost:6379/0"
redis_sync = sync_redis.from_url(REDIS_URL, decode_responses=True)


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Locust file content
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
    wait_time = between(1, 3)
    run_id: str | None = None

    @task(3)
    def submit_and_poll(self):
        query = QUERIES[int(time.time()) % len(QUERIES)]
        with self.client.post(
            "/agent/run",
            json={"query": query},
            catch_response=True,
            name="/agent/run",
        ) as resp:
            if resp.status_code != 202:
                resp.failure(f"Expected 202, got {resp.status_code}")
                return
            data = resp.json()
            self.run_id = data.get("run_id")

        if not self.run_id:
            return

        # Poll up to 5 times
        for _ in range(5):
            time.sleep(1)
            s = self.client.get(
                f"/agent/run/{self.run_id}/status",
                name="/agent/run/{id}/status",
            )
            if s.json().get("status") not in ("pending",):
                break

        # Fetch result
        self.client.get(
            f"/agent/run/{self.run_id}/result",
            name="/agent/run/{id}/result",
        )

    @task(1)
    def status_check(self):
        if self.run_id:
            self.client.get(
                f"/agent/run/{self.run_id}/status",
                name="/agent/run/{id}/status",
            )

    @task(1)
    def metrics_check(self):
        self.client.get("/metrics", name="/metrics")


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
    TTL = 300  # 5 minutes

    def _cache_key(self, query: str) -> str:
        return f"cache:query:{hashlib.sha256(query.encode()).hexdigest()}"

    async def get(self, query: str) -> dict | None:
        r = await aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            raw = await r.get(self._cache_key(query))
            return json.loads(raw) if raw else None
        finally:
            await r.aclose()

    async def set(self, query: str, result: dict):
        r = await aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await r.setex(self._cache_key(query), self.TTL, json.dumps(result))
        finally:
            await r.aclose()

    async def hit_rate(self) -> float:
        r = await aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            hits  = int(await r.get("cache:hits")  or 0)
            total = int(await r.get("cache:total") or 0)
            return hits / total if total > 0 else 0.0
        finally:
            await r.aclose()

    async def track_hit(self):
        r = await aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await r.incr("cache:hits")
            await r.incr("cache:total")
        finally:
            await r.aclose()

    async def track_miss(self):
        r = await aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            await r.incr("cache:total")
        finally:
            await r.aclose()


# ══════════════════════════════════════════════════════════════════════════════
# PART C — LLM Request Queue
# ══════════════════════════════════════════════════════════════════════════════

class LLMRequestQueue:
    def __init__(self, max_concurrent: int = 10):
        self._sem = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, *args):
        self._sem.release()


llm_queue = LLMRequestQueue(max_concurrent=10)


# ══════════════════════════════════════════════════════════════════════════════
# PART D — Per-User Rate Limiter
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RateLimitResult:
    allowed:     bool
    remaining:   int
    limit:       int
    retry_after: int


class RateLimiter:
    def __init__(self, requests_per_minute: int = 10):
        self.rpm    = requests_per_minute
        self.window = 60

    async def check(self, user_id: str) -> RateLimitResult:
        current_minute = int(time.time() // 60)
        key = f"ratelimit:{user_id}:{current_minute}"
        r   = await aioredis.from_url(REDIS_URL, decode_responses=True)
        try:
            pipe = r.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window + 5)
            count, _ = await pipe.execute()
            count = int(count)
        finally:
            await r.aclose()

        allowed     = count <= self.rpm
        remaining   = max(0, self.rpm - count)
        retry_after = (self.window - (int(time.time()) % self.window)) if not allowed else 0
        return RateLimitResult(allowed, remaining, self.rpm, retry_after)


# ══════════════════════════════════════════════════════════════════════════════
# PART E — Worker Auto-Scaling Simulator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScalingEvent:
    timestamp:    float
    queue_depth:  int
    worker_count: int
    action:       str


class WorkerScaler:
    def __init__(self, min_workers: int = 2, max_workers: int = 50):
        self.min_workers      = min_workers
        self.max_workers      = max_workers
        self._workers         = min_workers
        self._low_load_since: float | None = None
        self._history: list[ScalingEvent] = []

    def tick(self, queue_depth: int, timestamp: float | None = None) -> ScalingEvent:
        ts     = timestamp or time.time()
        action = "no_change"

        if queue_depth > 5 * self._workers and self._workers < self.max_workers:
            self._workers         = min(self._workers * 2, self.max_workers)
            action                = "scale_up"
            self._low_load_since  = None
        elif queue_depth < 2 * self._workers:
            if self._low_load_since is None:
                self._low_load_since = ts
            elif ts - self._low_load_since > 60 and self._workers > self.min_workers:
                self._workers        = max(self._workers // 2, self.min_workers)
                action               = "scale_down"
                self._low_load_since = None
        else:
            self._low_load_since = None

        event = ScalingEvent(ts, queue_depth, self._workers, action)
        self._history.append(event)
        return event

    def simulate(self, queue_depths: list[int], interval_s: float = 10.0) -> list[ScalingEvent]:
        events = []
        base   = time.time()
        for i, depth in enumerate(queue_depths):
            events.append(self.tick(depth, timestamp=base + i * interval_s))
        return events

    def render(self):
        changes = [e for e in self._history if e.action != "no_change"]
        if not changes:
            print("  No scaling events (load stable)")
        for e in changes:
            t = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            arrow = "⬆" if e.action == "scale_up" else "⬇"
            print(f"  {t}  {arrow} {e.action:<12}  queue={e.queue_depth:>3}  workers={e.worker_count}")
        final = self._history[-1] if self._history else None
        if final:
            print(f"  Final: {final.worker_count} workers (min={self.min_workers}, max={self.max_workers})")


# ══════════════════════════════════════════════════════════════════════════════
# PART F — Capacity Planner
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LoadTestResult:
    concurrent_users: int
    rps:              float
    p50_ms:           float
    p95_ms:           float
    error_rate_pct:   float
    worker_count:     int
    phase:            str = "after"


class CapacityPlanner:
    P95_SLA_MS     = 5_000
    ERROR_RATE_SLA = 1.0

    def max_supported_users(self, results: list[LoadTestResult]) -> int:
        passing = [r for r in results if r.p95_ms <= self.P95_SLA_MS and r.error_rate_pct <= self.ERROR_RATE_SLA]
        return max((r.concurrent_users for r in passing), default=0)

    def workers_needed(self, target_users: int) -> int:
        return math.ceil(target_users / 12)

    def monthly_cost(self, concurrent_users: int, rps: float, cost_per_request: float = 0.0018) -> float:
        return concurrent_users * rps * cost_per_request * 60 * 60 * 24 * 30

    def generate_report(self, results: list[LoadTestResult]):
        print(f"{'Users':>6}  {'RPS':>6}  {'P50ms':>7}  {'P95ms':>7}  {'Err%':>5}  {'Workers':>7}  SLA")
        print("─" * 60)
        for r in sorted(results, key=lambda x: x.concurrent_users):
            p95_ok  = r.p95_ms <= self.P95_SLA_MS
            err_ok  = r.error_rate_pct <= self.ERROR_RATE_SLA
            sla     = "✅ PASS" if (p95_ok and err_ok) else "❌ FAIL"
            print(f"  {r.concurrent_users:>4}  {r.rps:>6.1f}  {r.p50_ms:>7,.0f}  {r.p95_ms:>7,.0f}  {r.error_rate_pct:>5.1f}  {r.worker_count:>7}  {sla}")

        max_u = self.max_supported_users(results)
        print(f"\n  Max supported users:    {max_u}")
        print(f"  Workers for {max_u} users:  {self.workers_needed(max_u)}")
        best = next((r for r in sorted(results, key=lambda x: -x.concurrent_users) if r.concurrent_users == max_u), None)
        if best:
            cost = self.monthly_cost(max_u, best.rps)
            print(f"  Monthly LLM cost:       ${cost:,.2f}")
        print(f"  Scaling formula:        workers = ceil(users / 12)")


# ══════════════════════════════════════════════════════════════════════════════
# PART G — Before/After Comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_results(before: list[LoadTestResult], after: list[LoadTestResult]):
    before_map = {r.concurrent_users: r for r in before}
    after_map  = {r.concurrent_users: r for r in after}
    all_users  = sorted(set(before_map) | set(after_map))

    print(f"{'Users':>6}  {'P95 before':>11}  {'P95 after':>10}  {'P95 Δ':>8}  {'Err before':>11}  {'Err after':>10}  {'Err Δ':>8}")
    print("─" * 80)
    for u in all_users:
        b = before_map.get(u)
        a = after_map.get(u)
        if b and a:
            p95_imp = (b.p95_ms - a.p95_ms) / b.p95_ms * 100 if b.p95_ms else 0
            err_imp = b.error_rate_pct - a.error_rate_pct
            print(
                f"  {u:>4}  {b.p95_ms:>10,.0f}ms  {a.p95_ms:>9,.0f}ms  "
                f"{p95_imp:>+7.1f}%  {b.error_rate_pct:>10.1f}%  {a.error_rate_pct:>9.1f}%  {-err_imp:>+7.1f}pp"
            )
        elif a:
            print(f"  {u:>4}  {'—':>11}  {a.p95_ms:>9,.0f}ms  {'NEW':>8}  {'—':>11}  {a.error_rate_pct:>9.1f}%  {'NEW':>8}")


# ══════════════════════════════════════════════════════════════════════════════
# Mock data
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

MOCK_QUEUE = (
    [2]*6 + [15]*12 + [80]*18 + [5]*12 + [1]*6
)


async def main():
    print("=" * 60)
    print("Project 40 — Scale & Load Testing  (SOLUTION)")
    print("=" * 60)

    # Part B: Cache
    print("\n── Response Cache ──")
    cache = ResponseCache()
    test_query = "What are GDPR Article 28 requirements?"
    await cache.set(test_query, {"answer": "DPA required", "cost_usd": 0.0018})
    cached = await cache.get(test_query)
    print(f"  Cache hit: {cached is not None} {'✅' if cached else '❌'}")
    await cache.track_hit()
    await cache.track_miss()
    hit_rate = await cache.hit_rate()
    print(f"  Hit rate: {hit_rate:.0%}")

    # Part D: Rate limiter
    print("\n── Rate Limiter ──")
    limiter = RateLimiter(requests_per_minute=10)
    for i in range(12):
        result = await limiter.check("user-123")
        status = "✅ allowed" if result.allowed else f"❌ 429 (retry in {result.retry_after}s)"
        print(f"  Request {i+1:2d}: {status} | remaining={result.remaining}")

    # Part E: Auto-scaler
    print("\n── Worker Auto-Scaler ──")
    scaler = WorkerScaler(min_workers=2, max_workers=32)
    scaler.simulate(MOCK_QUEUE)
    scaler.render()

    # Part F: Capacity plan
    print("\n── Capacity Plan ──")
    planner = CapacityPlanner()
    planner.generate_report(MOCK_AFTER)

    # Part G: Before/After
    print("\n── Before vs After Fixes ──")
    compare_results(MOCK_BEFORE, MOCK_AFTER)

    print("\n── Locust Usage ──")
    print("  Save LOCUSTFILE_CONTENT as locustfile.py, start project39, then:")
    print("  locust -f locustfile.py --host=http://localhost:8000 --headless")
    print("         --users=100 --spawn-rate=10 --run-time=2m")


if __name__ == "__main__":
    asyncio.run(main())
