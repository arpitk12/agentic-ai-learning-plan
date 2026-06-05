"""
SOLUTION — Exercise 4: Load Testing with Locust

Run: locust -f sol4_load_test.py --host http://localhost:8000
     Then open: http://localhost:8089
"""
import os
import sys
import random
import time

try:
    from locust import HttpUser, task, between, events
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    print("Locust not installed. Run: pip install locust")

    class HttpUser:
        host = ""
        wait_time = None

    def task(weight=1):
        def decorator(fn): return fn
        return decorator

    def between(low, high): return None

    class events:
        @staticmethod
        def init(fn): return fn

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


class AgentUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    def on_start(self):
        self.session_id = f"load_test_{random.randint(1000, 9999)}"
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(5)
    def simple_question(self):
        question = random.choice(SIMPLE_QUESTIONS)
        with self.client.post(
            "/agent",
            json={"message": question, "session_id": self.session_id},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("answer"):
                    resp.failure("Empty answer")
            else:
                resp.failure(f"Expected 200, got {resp.status_code}")

    @task(2)
    def complex_question(self):
        question = random.choice(COMPLEX_QUESTIONS)
        with self.client.post(
            "/agent",
            json={"message": question, "session_id": self.session_id, "max_tokens": 512},
            catch_response=True,
            timeout=60,
        ) as resp:
            if resp.status_code == 200:
                pass  # success
            else:
                resp.failure(f"Expected 200, got {resp.status_code}")

    @task(3)
    def health_check(self):
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") != "healthy":
                    resp.failure(f"Unhealthy: {data}")
            else:
                resp.failure(f"Health check: {resp.status_code}")

    @task(1)
    def check_metrics(self):
        with self.client.get("/metrics", catch_response=True) as resp:
            if resp.status_code not in (200, 503):
                resp.failure(f"Metrics endpoint: {resp.status_code}")


class StreamingUser(HttpUser):
    wait_time = between(2, 5)
    host = "http://localhost:8000"

    @task(1)
    def streaming_request(self):
        question = random.choice(SIMPLE_QUESTIONS)
        with self.client.post(
            "/stream",
            json={"message": question, "session_id": f"stream_{random.randint(1000, 9999)}"},
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Stream: expected 200, got {resp.status_code}")
                return
            chunks_received = 0
            for line in resp.iter_lines():
                if line:
                    chunks_received += 1
            if chunks_received == 0:
                resp.failure("No SSE chunks received")


if LOCUST_AVAILABLE:
    @events.test_start.add_listener
    def on_test_start(environment, **kwargs):
        print("\n" + "="*60)
        print("LOAD TEST STARTING")
        print(f"Target: {environment.host}")
        print("="*60)

    @events.test_stop.add_listener
    def on_test_stop(environment, **kwargs):
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


def simple_load_test(host: str = "http://localhost:8000", num_requests: int = 20):
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
        print("\nRun:")
        print("  locust -f sol4_load_test.py --host http://localhost:8000")
        print("  # Then open http://localhost:8089")
        print("\nCI/headless:")
        print("  locust -f sol4_load_test.py --host http://localhost:8000 \\")
        print("    --users 10 --spawn-rate 2 --run-time 60s --headless --csv=results")
    else:
        print("Locust not installed — running simple load test...")
        simple_load_test()
