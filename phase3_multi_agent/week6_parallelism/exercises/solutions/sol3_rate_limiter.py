"""
SOLUTION — Exercise 3: Async Rate Limiter with Exponential Backoff
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import asyncio
import random
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm import achat, get_text

load_dotenv()

SIMULATE_ERRORS = True


@dataclass
class RateLimiter:
    max_per_minute: int = 10
    _timestamps: list[float] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self.max_per_minute:
                sleep_for = 60 - (now - self._timestamps[0])
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            self._timestamps.append(time.monotonic())


class SimulatedRateLimitError(Exception):
    pass


async def achat_maybe_fail(messages, **kwargs):
    if SIMULATE_ERRORS and random.random() < 0.25:
        raise SimulatedRateLimitError("429 Rate limit exceeded (simulated)")
    return await achat(messages, **kwargs)


async def call_with_backoff(messages: list[dict], idx: int,
                             max_retries: int = 4, base_delay: float = 1.0) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{idx}] attempt {attempt}", end=" ")
            response = await achat_maybe_fail(messages, max_tokens=80)
            print("→ OK")
            return get_text(response)
        except SimulatedRateLimitError as e:
            if attempt == max_retries:
                print(f"→ FAILED after {max_retries} retries")
                return f"[FAILED: {e}]"
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(-0.5, 0.5)
            delay = max(0.1, delay)
            print(f"→ RateLimit, retry in {delay:.1f}s")
            await asyncio.sleep(delay)
    return "[FAILED]"


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
    limiter = RateLimiter(max_per_minute=20)

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
