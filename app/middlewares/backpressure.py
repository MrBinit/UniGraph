import asyncio
import logging
import time
import uuid

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.infra.redis_client import app_redis_client, app_scoped_key

logger = logging.getLogger(__name__)


_ACQUIRE_BACKPRESSURE_SCRIPT = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local max_in_flight = tonumber(ARGV[2])
local lease_seconds = tonumber(ARGV[3])
local token = ARGV[4]

redis.call("ZREMRANGEBYSCORE", key, "-inf", now_ms)
local current = redis.call("ZCARD", key)
if current >= max_in_flight then
  local oldest = redis.call("ZRANGE", key, 0, 0, "WITHSCORES")
  local retry_after = 1
  if oldest[2] then
    local delta_ms = tonumber(oldest[2]) - now_ms
    retry_after = math.max(1, math.floor(delta_ms / 1000))
  end
  return {0, retry_after}
end

local expires_at = now_ms + (lease_seconds * 1000)
redis.call("ZADD", key, expires_at, token)
redis.call("EXPIRE", key, lease_seconds + 5)
return {1, 0}
"""


class _RedisBackpressureGate:
    """Distributed admission control gate for in-flight request capacity."""

    def __init__(self, *, key: str, max_in_flight_requests: int, lease_seconds: int):
        self.key = key.strip(": ")
        self.max_in_flight_requests = max_in_flight_requests
        self.lease_seconds = lease_seconds

    def acquire(self, token: str) -> tuple[bool, int]:
        now_ms = int(time.time() * 1000)
        result = app_redis_client.eval(
            _ACQUIRE_BACKPRESSURE_SCRIPT,
            1,
            self.key,
            now_ms,
            self.max_in_flight_requests,
            self.lease_seconds,
            token,
        )
        allowed = bool(result and int(result[0]) == 1)
        retry_after = int(result[1]) if isinstance(result, (list, tuple)) and len(result) > 1 else 1
        return allowed, max(1, retry_after)

    def release(self, token: str):
        app_redis_client.zrem(self.key, token)


class BackpressureMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        max_in_flight_requests: int,
        *,
        use_redis: bool = True,
        redis_key: str = "backpressure:inflight",
        distributed_lease_seconds: int = 45,
    ):
        super().__init__(app)
        self._max_in_flight_requests = max_in_flight_requests
        self._local_in_flight = 0
        self._local_gate_lock = asyncio.Lock()
        self._redis_gate = (
            _RedisBackpressureGate(
                key=app_scoped_key(redis_key),
                max_in_flight_requests=max_in_flight_requests,
                lease_seconds=distributed_lease_seconds,
            )
            if use_redis
            else None
        )

    async def _try_acquire_local_slot(self) -> bool:
        """Atomically reserve one local in-flight slot without private semaphore internals."""
        async with self._local_gate_lock:
            if self._local_in_flight >= self._max_in_flight_requests:
                return False
            self._local_in_flight += 1
            return True

    async def _release_local_slot(self):
        """Release one local in-flight slot."""
        async with self._local_gate_lock:
            self._local_in_flight = max(0, self._local_in_flight - 1)

    async def dispatch(self, request, call_next):
        redis_token = uuid.uuid4().hex
        acquired_distributed = False

        if self._redis_gate is not None:
            try:
                allowed, retry_after = await asyncio.to_thread(
                    self._redis_gate.acquire, redis_token
                )
            except Exception as exc:
                logger.warning(
                    "Distributed backpressure gate unavailable; falling back to local semaphore. %s",
                    exc,
                )
                allowed, retry_after = True, 0

            if not allowed:
                logger.warning(
                    "BackpressureRejectDistributed | method=%s path=%s retry_after=%s",
                    request.method,
                    request.url.path,
                    retry_after,
                )
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Server is busy. Please retry shortly."},
                    headers={"Retry-After": str(retry_after)},
                )
            acquired_distributed = True

        acquired_local = await self._try_acquire_local_slot()
        if not acquired_local:
            logger.warning(
                "BackpressureReject | method=%s path=%s",
                request.method,
                request.url.path,
            )
            if acquired_distributed and self._redis_gate is not None:
                try:
                    await asyncio.to_thread(self._redis_gate.release, redis_token)
                except Exception:
                    logger.warning(
                        "Failed releasing distributed backpressure token after local reject."
                    )
            return JSONResponse(
                status_code=503,
                content={"detail": "Server is busy. Please retry shortly."},
            )

        try:
            return await call_next(request)
        finally:
            await self._release_local_slot()
            if acquired_distributed and self._redis_gate is not None:
                try:
                    await asyncio.to_thread(self._redis_gate.release, redis_token)
                except Exception:
                    logger.warning("Failed releasing distributed backpressure token.")
