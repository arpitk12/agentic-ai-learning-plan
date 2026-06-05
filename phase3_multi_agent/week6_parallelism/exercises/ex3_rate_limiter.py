"""
Exercise 3: Async Rate Limiter with Exponential Backoff
Goal: Handle API rate limits gracefully in async agents.

Tasks:
  1. Complete RateLimiter.acquire() — track requests per minute, sleep if needed.
  2. Complete call_with_backoff() — wrap achat() with exponential backoff on
     RateLimitError (max 4 retries, base delay 1s, jitter ±0.5s).
  3. Complete fan_out_rate_limited() — run N calls through the rate limiter.
  4. Simulate rate limit errors randomly (20% chance) via SIMULATE_ERRORS=True.
  5. Print a summary: total calls, retries, wall time, effective calls/min.

Expected output:
  [1] attempt 1 → OK
  [2] attempt 1 → RateLimit, retry in 1.3s
  [2] attempt 2 → OK
  ...
  Summary: 10 calls | 2 retries | 8.1s | 74 calls/min
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import asyncio
import random
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

SIMULATE_ERRORS = True   # set False to use real API without errors

# ── Rate Limiter ───────────────────────────────────────────────────────────────

@dataclass
class RateLimiter:
    max_per_minute: int = 10
    _timestamps: list[float] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self):
        """Block until a request slot is available within the rate limit window."""
        # TODO: async with self._lock:
        # TODO:   now = time.monotonic()
        # TODO:   # Remove timestamps older than 60 seconds
        # TODO:   self._timestamps = [t for t in self._timestamps if now - t < 60]
        # TODO:   if len(self._timestamps) >= self.max_per_minute:
        # TODO:       sleep_for = 60 - (now - self._timestamps[0])
        # TODO:       await asyncio.sleep(max(0, sleep_for))
        # TODO:   self._timestamps.append(time.monotonic())
        raise NotImplementedError


# ── Simulated Rate Limit Error ─────────────────────────────────────────────────

class SimulatedRateLimitError(Exception):
    pass


async def achat_maybe_fail(messages, **kwargs):
    """Wrap achat to simulate occasional rate limit errors."""
    if SIMULATE_ERRORS and random.random() < 0.25:
        raise SimulatedRateLimitError("429 Rate limit exceeded (simulated)")
    return await achat(messages, **kwargs)


# ── Exponential Backoff ────────────────────────────────────────────────────────

async def call_with_backoff(messages: list[dict], idx: int,
                             max_retries: int = 4, base_delay: float = 1.0) -> str:
    """
    Call achat with exponential backoff on rate limit errors.
    Delay = base_delay * 2^attempt + random jitter in [-0.5, 0.5]
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{idx}] attempt {attempt}", end=" ")
            # TODO: response = await achat_maybe_fail(messages, max_tokens=80)
            # TODO: print("→ OK")
            # TODO: return get_text(response)
            raise NotImplementedError
        except SimulatedRateLimitError as e:
            if attempt == max_retries:
                print(f"→ FAILED after {max_retries} retries")
                return f"[FAILED: {e}]"
            # TODO: delay = base_delay * (2 ** (attempt - 1)) + random.uniform(-0.5, 0.5)
            # TODO: delay = max(0.1, delay)
            # TODO: print(f"→ RateLimit, retry in {delay:.1f}s")
            # TODO: await asyncio.sleep(delay)
            raise NotImplementedError


# ── Fan-out with Rate Limiter ──────────────────────────────────────────────────

PROMPTS = [
    "Name one benefit of async programming.",
    "What is a semaphore used for?",
    "Name one downside of rate limiting.",
    "Define exponential backoff in one sentence.",
    "What does asyncio.gather do?",
    "Name a Python async HTTP library.",
    "What is jitter in retry logic?",
    "Define throughput in one sentence.",
    "What is a token bucket algorithm?",
    "Name one advantage of fan-out parallelism.",
]


async def fan_out_rate_limited(prompts: list[str]) -> list[str]:
    """Run all prompts with rate limiting + backoff. Track retry count."""
    limiter = RateLimiter(max_per_minute=20)
    results = []
    total_retries = [0]

    async def run_one(prompt: str, idx: int) -> str:
        await limiter.acquire()
        messages = [{"role": "user", "content": prompt}]
        return await call_with_backoff(messages, idx)

    t0 = time.perf_counter()
    tasks = [run_one(p, i + 1) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    calls_per_min = len(prompts) / elapsed * 60
    print(f"\n{'─'*40}")
    print(f"Summary: {len(prompts)} calls | {elapsed:.1f}s | {calls_per_min:.0f} calls/min")
    return list(results)


if __name__ == "__main__":
    results = asyncio.run(fan_out_rate_limited(PROMPTS))
    print("\nResults:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r[:80]}")
