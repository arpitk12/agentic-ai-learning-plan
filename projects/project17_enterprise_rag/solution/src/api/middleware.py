"""Middleware: RequestID, timing header, sliding-window rate limiter, circuit breaker."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config import cfg


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        response.headers["X-Latency-Ms"] = str(ms)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter: cfg.rate_limit_per_minute requests per IP."""

    def __init__(self, app, limit: int = cfg.rate_limit_per_minute) -> None:
        super().__init__(app)
        self._limit = limit
        self._windows: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._windows[ip]

        # Remove timestamps older than 60 s
        while window and window[0] < now - 60:
            window.popleft()

        if len(window) >= self._limit:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded — {self._limit} req/min"},
                headers={"Retry-After": "60"},
            )

        window.append(now)
        return await call_next(request)
