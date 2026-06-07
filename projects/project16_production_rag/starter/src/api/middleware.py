"""
TODO — Implement request middleware.

Every HTTP request should get:
  1. A unique X-Request-ID header (UUID4, first 8 chars is fine)
     → Store on request.state.request_id so routes can log it
  2. A X-Latency-Ms response header (time.perf_counter diff in ms)
  3. A structured log line: METHOD PATH → STATUS  latency ms  [request_id]

Also implement a simple in-memory rate limiter:
  - Track requests per IP using a dict of deques (timestamps in last 60s)
  - If count >= cfg.RATE_LIMIT, return 429 immediately
  - Skip rate limiting for /health (liveness probes must always pass)

Structure (Starlette middleware):
  class RequestMiddleware(BaseHTTPMiddleware):
      async def dispatch(self, request, call_next) -> Response:
          ...
"""
from __future__ import annotations
import time
import uuid
import logging
from collections import defaultdict, deque
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from src.config import cfg

logger = logging.getLogger(__name__)


class RequestMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit: int = cfg.RATE_LIMIT):
        super().__init__(app)
        self._rate_limit = rate_limit
        self._windows: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        TODO 1: Generate request_id = str(uuid.uuid4())[:8]
        TODO 2: request.state.request_id = request_id
        TODO 3: Record t0 = time.perf_counter()

        TODO 4: If request.url.path != "/health":
                  client_ip = request.client.host (or "unknown")
                  window = self._windows[client_ip]
                  Drop timestamps older than 60s from the front of window
                  If len(window) >= self._rate_limit:
                      return Response(content=JSON 429 message, status_code=429,
                                      headers={"X-Request-ID": request_id})
                  window.append(time.time())

        TODO 5: response = await call_next(request)
        TODO 6: latency = round((time.perf_counter() - t0) * 1000, 1)
        TODO 7: response.headers["X-Request-ID"] = request_id
        TODO 8: response.headers["X-Latency-Ms"] = str(latency)
        TODO 9: logger.info("%s %s → %d  %.1fms  [%s]", method, path, status, latency, rid)
        TODO 10: return response
        """
        raise NotImplementedError
