"""
Exercise 4: Load Testing with Locust
Goal: Stress-test your agent API to find throughput limits and bottlenecks.

Install: pip install locust

What Locust measures:
  - Requests per second (RPS) — throughput
  - Response time P50/P95/P99 — latency percentiles
  - Failure rate — % of requests that error
  - Users — concurrent simulated users

Run:
  # Web UI mode (recommended for exploration):
  locust -f ex4_load_test.py --host http://localhost:8000

  # Headless / CI mode:
  locust -f ex4_load_test.py --host http://localhost:8000 \\
    --users 10 --spawn-rate 2 --run-time 60s --headless \\
    --csv=load_test_results

  # Then open http://localhost:8089 in your browser for the Locust UI.

Tasks:
  1. Review AgentUser tasks — understand what each @task simulates.
  2. Complete the chat_with_agent() task — POST to /agent, assert status 200.
  3. Complete the streaming_request() task — GET /stream, consume SSE chunks.
  4. Add a @task(weight=1) for a heavy multi-tool request.
  5. Run against your FastAPI server and identify the bottleneck.
  6. (Bonus) Write a custom Locust event hook to log P95 latency every 10s.

Expected findings:
  - The LLM API call is almost always the bottleneck (~80-95% of latency).
  - Increasing workers helps for concurrent users but not per-request speed.
  - Streaming endpoints feel faster to users even if total time is the same.
"""
import os
import sys
import json
import random
import time

# Locust import (required for test discovery)
try:
    from locust import HttpUser, task, between, events
    from locust.runners import MasterRunner, WorkerRunner
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    print("Locust not installed. Run: pip install locust")

    # Stub classes so the rest of the file is importable
    class HttpUser:
        host = ""
        wait_time = None

    def task(weight=1):
        def decorator(fn):
            return fn
        return decorator

    def between(low, high):
        return None

    class events:
        @staticmethod
        def init(fn): return fn


# ── Test Data ─────────────────────────────────────────────────────────────────

SIMPLE_QUESTIONS = [
    "What is 42 times 7?",
    "What is the capital of Japan?",
    "How many bytes are in a kilobyte?",
    "What year was Python first released?",
    "What does API stand for?",
]

COMPLEX_QUESTIONS = [
    "Calculate the compound interest on $10,000 at 5% annual rate for 3 years.",
    "What are the top 3 benefits of using async programming in Python?",
    "Explain the difference between SQL and NoSQL databases in 2 sentences.",
    "What is the time complexity of binary search?",
]

MATH_EXPRESSIONS = [
    "100 * 42 + 17",
    "(2 ** 10) - 1",
    "sum(range(100))",
    "round(3.14159 * 2, 4)",
]


# ── User Classes ──────────────────────────────────────────────────────────────

class AgentUser(HttpUser):
    """
    Simulates a typical user of the agent API.
    wait_time: users wait 1-3 seconds between requests (realistic pacing).
    """
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    def on_start(self):
        """Called once when this user starts. Use for login/setup."""
        self.session_id = f"load_test_{random.randint(1000, 9999)}"
        # Health check to verify server is up
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(5)  # weight 5 — most common request
    def simple_question(self):
        """
        Send a simple factual question to the /agent endpoint.

        TODO:
        1. Pick a random question from SIMPLE_QUESTIONS.
        2. POST to /agent with JSON body: {"message": question, "session_id": self.session_id}
        3. Use self.client.post(..., catch_response=True) for proper error handling.
        4. If status != 200, call resp.failure(f"Expected 200, got {resp.status_code}").
        5. Optionally verify the answer is non-empty.
        """
        raise NotImplementedError

    @task(2)  # weight 2 — less frequent
    def complex_question(self):
        """
        Send a complex question that may trigger longer LLM reasoning.

        TODO:
        1. Pick a random question from COMPLEX_QUESTIONS.
        2. POST to /agent with max_tokens=512.
        3. Record failure if status != 200 or latency > 30s.
        """
        raise NotImplementedError

    @task(3)  # weight 3
    def health_check(self):
        """
        Poll the health endpoint — simulates monitoring/load-balancer checks.
        This is already implemented as a reference for the other tasks.
        """
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") != "healthy":
                    resp.failure(f"Unhealthy: {data}")
            else:
                resp.failure(f"Health check: {resp.status_code}")

    @task(1)  # weight 1 — rare heavy request
    def check_metrics(self):
        """Fetch Prometheus metrics — simulates a monitoring scraper."""
        with self.client.get("/metrics", catch_response=True) as resp:
            if resp.status_code not in (200, 503):
                resp.failure(f"Metrics endpoint: {resp.status_code}")


class StreamingUser(HttpUser):
    """
    Simulates users using the streaming /stream endpoint.
    Lower wait time — streaming users tend to be more interactive.
    """
    wait_time = between(2, 5)
    host = "http://localhost:8000"

    @task(1)
    def streaming_request(self):
        """
        Send a streaming request and consume SSE chunks.

        TODO:
        1. POST to /stream with {"message": <random question>, "session_id": ...}
        2. Set stream=True in the requests call to get chunks incrementally.
        3. Count chunks received.
        4. Fail if no chunks received or status != 200.

        Hint: self.client.post("/stream", json=..., stream=True)
              then iterate resp.iter_lines() or resp.iter_content()
        """
        raise NotImplementedError


# ── Custom Locust Events ──────────────────────────────────────────────────────

if LOCUST_AVAILABLE:
    @events.test_start.add_listener
    def on_test_start(environment, **kwargs):
        print("\n" + "="*60)
        print("LOAD TEST STARTING")
        print(f"Target: {environment.host}")
        print("="*60)

    @events.test_stop.add_listener
    def on_test_stop(environment, **kwargs):
        """Print a final summary when the test ends."""
        stats = environment.runner.stats
        print("\n" + "="*60)
        print("LOAD TEST COMPLETE — SUMMARY")
        print("="*60)

        for name, stat in stats.entries.items():
            if stat.num_requests == 0:
                continue
            print(f"\n  Endpoint: {name}")
            print(f"    Requests: {stat.num_requests}")
            print(f"    Failures: {stat.num_failures} ({stat.fail_ratio:.1%})")
            print(f"    RPS:      {stat.current_rps:.1f}")
            print(f"    P50:      {stat.get_response_time_percentile(0.50):.0f}ms")
            print(f"    P95:      {stat.get_response_time_percentile(0.95):.0f}ms")
            print(f"    P99:      {stat.get_response_time_percentile(0.99):.0f}ms")

    # TODO (Bonus): Add a periodic stats logger
    # @events.request.add_listener
    # def on_request(request_type, name, response_time, response_length, **kwargs):
    #     # Log every request to a CSV for offline analysis
    #     pass


# ── Headless Runner (no Locust) ───────────────────────────────────────────────

def simple_load_test(host: str = "http://localhost:8000", num_requests: int = 20):
    """
    Basic load test using only requests library — no Locust needed.
    Useful for quick benchmarks.
    """
    import requests
    import statistics

    print(f"Simple load test: {num_requests} requests to {host}/health")
    latencies = []
    errors = 0

    for i in range(num_requests):
        t0 = time.monotonic()
        try:
            r = requests.get(f"{host}/health", timeout=10)
            if r.status_code != 200:
                errors += 1
        except Exception:
            errors += 1
        latencies.append((time.monotonic() - t0) * 1000)
        print(f"  [{i+1:3d}/{num_requests}] {latencies[-1]:.0f}ms", end="\r")

    print(f"\n\nResults:")
    print(f"  Requests:  {num_requests}")
    print(f"  Errors:    {errors} ({errors/num_requests:.0%})")
    print(f"  P50:       {statistics.median(latencies):.0f}ms")
    print(f"  P95:       {sorted(latencies)[int(0.95*len(latencies))]:.0f}ms")
    print(f"  Max:       {max(latencies):.0f}ms")


if __name__ == "__main__":
    if LOCUST_AVAILABLE:
        print("Locust is installed.")
        print("\nRun with:")
        print("  locust -f ex4_load_test.py --host http://localhost:8000")
        print("  # Then open http://localhost:8089 in your browser")
        print("\nCI/headless mode:")
        print("  locust -f ex4_load_test.py --host http://localhost:8000 \\")
        print("    --users 10 --spawn-rate 2 --run-time 60s --headless --csv=results")
    else:
        print("Locust not installed — running simple load test instead...")
        simple_load_test()
