"""Middleware: RequestID, timing, rate limiter. See solution for full implementation."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config import cfg


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    TODO 1: In dispatch(), read X-Request-ID header (or generate uuid4()).
            Set request.state.request_id = rid.
            Call response = await call_next(request).
            Add X-Request-ID to response headers.
            Return response.
    """
    async def dispatch(self, request: Request, call_next):
        raise NotImplementedError


class TimingMiddleware(BaseHTTPMiddleware):
    """
    TODO 2: Record t0 = time.perf_counter() before call_next.
            Add X-Latency-Ms header to response.
    """
    async def dispatch(self, request: Request, call_next):
        raise NotImplementedError


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    TODO 3: Implement a sliding-window rate limiter.
            Pass-through /health and /metrics without rate limiting.
            Use a defaultdict(deque) keyed by client IP.
            For each request: remove timestamps > 60s old, check count vs cfg.rate_limit_per_minute.
            Return 429 if over limit, else append current timestamp and continue.
    """
    def __init__(self, app, limit: int = cfg.rate_limit_per_minute) -> None:
        super().__init__(app)
        self._limit = limit
        self._windows: dict = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        raise NotImplementedError
