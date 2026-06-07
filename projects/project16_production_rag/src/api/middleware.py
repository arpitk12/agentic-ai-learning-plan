"""
Request/response middleware:
  - Attaches a unique X-Request-ID header to every request
  - Logs every request with method, path, status, latency (structlog JSON)
  - Simple in-memory rate limiter (requests/minute per IP)
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
    """Attach request ID, log every request, enforce rate limit."""

    def __init__(self, app, rate_limit: int = cfg.RATE_LIMIT):
        super().__init__(app)
        self._rate_limit = rate_limit
        self._windows: dict[str, deque] = defaultdict(deque)   # IP → timestamps

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        t0         = time.perf_counter()

        # Rate limit check (skip for /health)
        if request.url.path != "/health":
            client_ip = request.client.host if request.client else "unknown"
            now       = time.time()
            window    = self._windows[client_ip]
            # Drop entries older than 60s
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self._rate_limit:
                logger.warning("Rate limit exceeded: %s", client_ip)
                return Response(
                    content='{"detail":"Rate limit exceeded. Max %d req/min."}' % self._rate_limit,
                    status_code=429,
                    media_type="application/json",
                    headers={"X-Request-ID": request_id},
                )
            window.append(now)

        response = await call_next(request)
        latency  = round((time.perf_counter() - t0) * 1000, 1)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(latency)

        logger.info(
            "%s %s → %d  %.1fms  [%s]",
            request.method, request.url.path,
            response.status_code, latency, request_id,
        )
        return response
