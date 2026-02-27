import asyncio
import logging
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_seconds: int):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            request_id = getattr(request.state, "request_id", "")
            logger.warning(
                "RequestTimeout | request_id=%s method=%s path=%s timeout_seconds=%s",
                request_id,
                request.method,
                request.url.path,
                self.timeout_seconds,
            )
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out."},
            )
