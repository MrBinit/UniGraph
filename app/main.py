import logging

from fastapi import FastAPI

from app.api.v1.chat import router as chat_router
from app.api.v1.ops import router as ops_router
from app.core.config import get_settings
from app.middlewares.backpressure import BackpressureMiddleware
from app.middlewares.rate_limit import RateLimitMiddleware
from app.middlewares.request_logging import RequestLoggingMiddleware
from app.middlewares.route_matching import RouteMatchingMiddleware
from app.middlewares.timeout import TimeoutMiddleware

settings = get_settings()


def _configure_logging():
    level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(title=settings.app.name)

    if settings.middleware.enable_route_matching:
        app.add_middleware(RouteMatchingMiddleware)
    if settings.middleware.enable_backpressure:
        app.add_middleware(
            BackpressureMiddleware,
            max_in_flight_requests=settings.middleware.max_in_flight_requests,
        )
    if settings.middleware.enable_timeout:
        app.add_middleware(
            TimeoutMiddleware,
            timeout_seconds=settings.middleware.timeout_seconds,
        )
    if settings.middleware.enable_rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            limit=settings.middleware.rate_limit_requests,
            window_seconds=settings.middleware.rate_limit_window_seconds,
        )
    if settings.middleware.enable_request_logging:
        app.add_middleware(RequestLoggingMiddleware)

    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(ops_router, prefix="/api/v1")
    return app


app = create_app()
