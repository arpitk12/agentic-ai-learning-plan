"""
SOLUTION — Exercise 4: Per-User Rate Limiter Middleware
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except ImportError:
    raise SystemExit("Run: pip install fastapi uvicorn")

from llm import chat, get_text

app = FastAPI(title="Rate-Limited Agent API")


@dataclass
class TokenBucket:
    capacity: float = 5.0
    refill_rate: float = 0.1
    tokens: float = field(init=False)
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.tokens = self.capacity

    def consume(self) -> tuple[bool, float]:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        retry_after = (1.0 - self.tokens) / self.refill_rate
        return False, retry_after

    def remaining(self) -> float:
        now = time.monotonic()
        elapsed = now - self.last_refill
        return min(self.capacity, self.tokens + elapsed * self.refill_rate)


BUCKETS: dict[str, TokenBucket] = {}


def get_bucket(user_id: str) -> TokenBucket:
    if user_id not in BUCKETS:
        BUCKETS[user_id] = TokenBucket()
    return BUCKETS[user_id]


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    user_id = request.query_params.get("user", "anonymous")
    path = request.url.path

    if path in {"/health", "/docs", "/openapi.json"} or path.startswith("/admin"):
        return await call_next(request)

    bucket = get_bucket(user_id)
    allowed, retry_after = bucket.consume()

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": round(retry_after, 1)},
            headers={
                "X-RateLimit-Limit": str(int(bucket.capacity)),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(int(retry_after)),
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(int(bucket.capacity))
    response.headers["X-RateLimit-Remaining"] = str(int(bucket.remaining()))
    return response


@app.get("/agent")
async def agent_endpoint(query: str = "What is 2+2?", user: str = "anonymous"):
    response = chat([{"role": "user", "content": query}], max_tokens=128)
    return {"user": user, "query": query, "answer": get_text(response)}


@app.get("/admin/limits")
async def admin_limits():
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
