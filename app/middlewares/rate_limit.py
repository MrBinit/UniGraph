import logging
import time
from collections import defaultdict, deque
from threading import Lock
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class _InMemorySlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._events = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            bucket = self._events[key]
            cutoff = now - self.window_seconds
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                return False, retry_after

            bucket.append(now)
            return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int, window_seconds: int):
        super().__init__(app)
        self._limiter = _InMemorySlidingWindowLimiter(limit=limit, window_seconds=window_seconds)

    @staticmethod
    def _client_key(request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            first = forwarded.split(",")[0].strip()
            if first:
                return f"{first}:{request.url.path}"
        host = request.client.host if request.client else "unknown"
        return f"{host}:{request.url.path}"

    async def dispatch(self, request, call_next):
        key = self._client_key(request)
        allowed, retry_after = self._limiter.allow(key)
        if not allowed:
            logger.warning("RateLimitExceeded | key=%s retry_after=%s", key, retry_after)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
