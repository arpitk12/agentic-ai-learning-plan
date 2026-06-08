"""
solution/src/api/middleware.py — Full implementation.
"""
from __future__ import annotations
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore
from starlette.responses import JSONResponse  # type: ignore


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rpm: int = 60):
        super().__init__(app)
        self._rpm = rpm
        self._window = 60.0
        self._store: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Prune old timestamps
        recent = [t for t in self._store[ip] if t > now - self._window]

        if len(recent) >= self._rpm:
            wait = self._window - (now - recent[0]) if recent else self._window
            return JSONResponse(
                {"error": "rate limit exceeded", "retry_after_seconds": round(wait)},
                status_code=429,
                headers={"Retry-After": str(int(wait))},
            )

        recent.append(now)
        self._store[ip] = recent

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._rpm)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self._rpm - len(recent)))
        return response
