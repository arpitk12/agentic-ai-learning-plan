"""
Request middleware: attaches a unique request ID, measures latency, logs every
request, and enforces a per-IP rate limit using a sliding 60-second window.
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
        TODO 1: Generate a short unique request ID and attach it to request.state
        TODO 2: Record the start time
        TODO 3: For non-health requests, check the per-IP sliding window rate limit;
                return 429 immediately if exceeded, otherwise record the current timestamp
        TODO 4: Call the next handler to get the response
        TODO 5: Compute latency and add X-Request-ID and X-Latency-Ms response headers
        TODO 6: Log the method, path, status code, latency, and request ID
        TODO 7: Return the response
        """
        raise NotImplementedError
