"""
src/api/middleware.py — Rate limiting middleware (sliding window, in-memory).

TODO:
  1. implement RateLimitMiddleware using Starlette BaseHTTPMiddleware
     - sliding window per client IP
     - 429 response with Retry-After header when limit exceeded
"""
from __future__ import annotations
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


# ── TODO 1: Rate limit middleware ─────────────────────────────────────────────
# class RateLimitMiddleware(BaseHTTPMiddleware):
#     """
#     Sliding window rate limiter per client IP.
#
#     Args:
#         rpm: Max requests per minute per IP.
#     """
#     def __init__(self, app, rpm: int = 60):
#         super().__init__(app)
#         self._rpm = rpm
#         self._window = 60.0   # seconds
#         self._store: dict[str, list[float]] = defaultdict(list)
#
#     async def dispatch(self, request: Request, call_next):
#         """
#         Steps:
#           1a. client_ip = request.client.host
#           1b. now = time.time()
#           1c. Prune old timestamps: [t for t in self._store[ip] if t > now - window]
#           1d. If len(recent) >= rpm:
#                   remaining_wait = window - (now - recent[0])
#                   return JSONResponse(
#                       {"error": "rate limit exceeded", "retry_after": remaining_wait},
#                       status_code=429,
#                       headers={"Retry-After": str(int(remaining_wait))},
#                   )
#           1e. Append now to self._store[ip]
#           1f. response = await call_next(request)
#           1g. Add headers: X-RateLimit-Limit, X-RateLimit-Remaining
#           1h. Return response
#         """
#         raise NotImplementedError

raise NotImplementedError("Implement RateLimitMiddleware in src/api/middleware.py")
