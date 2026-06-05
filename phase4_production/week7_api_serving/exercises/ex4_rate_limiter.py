"""
Exercise 4: Per-User Rate Limiter Middleware
Goal: Enforce request limits per user to prevent abuse and control costs.

Install: pip install fastapi uvicorn

Run: uvicorn ex4_rate_limiter:app --reload --port 8002
Test:
  # Should succeed (first 5 requests)
  for i in $(seq 1 5); do curl -s "http://localhost:8002/agent?query=hi&user=alice" | python3 -m json.tool; done
  # Should get 429 on 6th
  curl -s "http://localhost:8002/agent?query=hi&user=alice" | python3 -m json.tool

Tasks:
  1. Complete TokenBucket.consume() — allow request if tokens remain, replenish over time.
  2. Complete RateLimiterMiddleware — intercept requests, check bucket, return 429 if empty.
  3. Add /admin/limits endpoint to view current bucket state for all users.
  4. (Bonus) Add global rate limit (across all users) in addition to per-user.

Expected headers on 429:
  HTTP/1.1 429 Too Many Requests
  X-RateLimit-Limit: 5
  X-RateLimit-Remaining: 0
  Retry-After: 12
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import time
import asyncio
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
except ImportError:
    raise SystemExit("Run: pip install fastapi uvicorn")

from llm import chat, get_text

app = FastAPI(title="Rate-Limited Agent API")

# ── Token Bucket ───────────────────────────────────────────────────────────────

@dataclass
class TokenBucket:
    """Token bucket algorithm: users get `capacity` tokens, refilled at `refill_rate/s`."""
    capacity: float = 5.0          # max requests per window
    refill_rate: float = 0.1       # tokens per second (1 per 10s)
    tokens: float = field(init=False)
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.tokens = self.capacity

    def consume(self) -> tuple[bool, float]:
        """
        Try to consume 1 token. Return (allowed, retry_after_seconds).
        TODO:
          1. Compute elapsed = time.monotonic() - self.last_refill
          2. Add elapsed * self.refill_rate tokens (cap at capacity)
          3. Update self.last_refill
          4. If self.tokens >= 1: subtract 1, return (True, 0)
          5. Else: return (False, (1 - self.tokens) / self.refill_rate)
        """
        raise NotImplementedError

    def remaining(self) -> float:
        now = time.monotonic()
        elapsed = now - self.last_refill
        return min(self.capacity, self.tokens + elapsed * self.refill_rate)


# ── Per-User Registry ──────────────────────────────────────────────────────────

BUCKETS: dict[str, TokenBucket] = {}

def get_bucket(user_id: str) -> TokenBucket:
    if user_id not in BUCKETS:
        BUCKETS[user_id] = TokenBucket()
    return BUCKETS[user_id]


# ── Rate Limit Middleware ──────────────────────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    TODO:
      1. Extract user_id from query param 'user' (default 'anonymous').
      2. Skip rate limiting for /health and /admin paths.
      3. Get (or create) TokenBucket for user_id.
      4. Call bucket.consume().
      5. If not allowed: return JSONResponse(status_code=429, ...) with headers:
           X-RateLimit-Limit: {capacity}
           X-RateLimit-Remaining: 0
           Retry-After: {retry_after:.0f}
      6. If allowed: proceed with call_next(request), add remaining header.
    """
    user_id = request.query_params.get("user", "anonymous")
    path = request.url.path

    # Skip non-agent paths
    if path in {"/health", "/docs", "/openapi.json"} or path.startswith("/admin"):
        return await call_next(request)

    bucket = get_bucket(user_id)
    # TODO: implement rate limit check
    raise NotImplementedError


# ── Agent Endpoint ─────────────────────────────────────────────────────────────

@app.get("/agent")
async def agent_endpoint(query: str = "What is 2+2?", user: str = "anonymous"):
    """Simple agent endpoint — rate-limited by middleware."""
    response = chat([{"role": "user", "content": query}], max_tokens=128)
    return {"user": user, "query": query, "answer": get_text(response)}


@app.get("/admin/limits")
async def admin_limits():
    """Show current rate limit state for all users."""
    return {
        user_id: {
            "tokens_remaining": round(bucket.remaining(), 2),
            "capacity": bucket.capacity,
            "refill_rate_per_sec": bucket.refill_rate,
        }
        for user_id, bucket in BUCKETS.items()
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=False)
    # Test: curl "http://localhost:8002/agent?query=hi&user=alice"
