import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "RequestError | request_id=%s method=%s path=%s latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        user_id = getattr(request.state, "user_id", "anonymous")
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            "RequestLog | request_id=%s method=%s path=%s status=%s latency_ms=%.2f user_id=%s client_ip=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            user_id,
            client_ip,
        )
        response.headers["X-Request-ID"] = request_id
        return response
