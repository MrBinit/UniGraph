import asyncio
import logging
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class BackpressureMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_in_flight_requests: int):
        super().__init__(app)
        self._semaphore = asyncio.Semaphore(max_in_flight_requests)

    async def dispatch(self, request, call_next):
        if self._semaphore._value <= 0:  # noqa: SLF001
            logger.warning(
                "BackpressureReject | method=%s path=%s",
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=503,
                content={"detail": "Server is busy. Please retry shortly."},
            )

        await self._semaphore.acquire()
        try:
            return await call_next(request)
        finally:
            self._semaphore.release()
