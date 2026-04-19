import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.evaluation import router as evaluation_router
from app.api.v1.ops import router as ops_router
from app.core.config import get_settings
from app.core.security import validate_security_configuration
from app.middlewares.backpressure import BackpressureMiddleware
from app.middlewares.rate_limit import RateLimitMiddleware
from app.middlewares.request_logging import RequestLoggingMiddleware
from app.middlewares.route_matching import RouteMatchingMiddleware
from app.middlewares.timeout import TimeoutMiddleware
from app.infra.redis_client import app_redis_client
from app.services.offline_evaluation_service import (
    start_offline_eval_scheduler,
    stop_offline_eval_scheduler,
)

settings = get_settings()
logger = logging.getLogger(__name__)
API_V1_PREFIX = "/api/v1"
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"


def _configure_logging():
    """Initialize application logging from the configured log level."""
    level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _warm_redis_backend():
    """Warm Redis connectivity to reduce first-request latency."""
    try:
        app_redis_client.ping()
        logger.info("StartupWarmup | redis=ok")
    except Exception as exc:
        logger.warning("StartupWarmup | redis=failed | error=%s", exc)


def _register_lifecycle_handlers(app: FastAPI):
    """Register startup/shutdown handlers."""

    def warm_backends():
        _warm_redis_backend()
        start_offline_eval_scheduler()

    async def stop_background_jobs():
        await stop_offline_eval_scheduler()

    app.add_event_handler("startup", warm_backends)
    app.add_event_handler("shutdown", stop_background_jobs)


def _add_enabled_middlewares(app: FastAPI):
    """Attach configured middlewares."""
    if settings.middleware.enable_route_matching:
        app.add_middleware(RouteMatchingMiddleware)
    if settings.middleware.enable_backpressure:
        app.add_middleware(
            BackpressureMiddleware,
            max_in_flight_requests=settings.middleware.max_in_flight_requests,
            use_redis=settings.middleware.enable_distributed_backpressure,
            redis_key=settings.middleware.distributed_backpressure_key,
            distributed_lease_seconds=settings.middleware.distributed_backpressure_lease_seconds,
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
            use_redis=settings.middleware.enable_distributed_rate_limit,
            redis_key_prefix=settings.middleware.distributed_rate_limit_prefix,
            trusted_proxy_cidrs=settings.middleware.trusted_proxy_cidrs,
        )
    if settings.middleware.enable_request_logging:
        app.add_middleware(RequestLoggingMiddleware)
    if settings.middleware.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.middleware.cors_allow_origins,
            allow_methods=settings.middleware.cors_allow_methods,
            allow_headers=settings.middleware.cors_allow_headers,
            allow_credentials=settings.middleware.cors_allow_credentials,
        )


def _include_api_routers(app: FastAPI):
    """Register API routes."""
    app.include_router(chat_router, prefix=API_V1_PREFIX)
    app.include_router(auth_router, prefix=API_V1_PREFIX)
    app.include_router(evaluation_router, prefix=API_V1_PREFIX)
    app.include_router(ops_router, prefix=API_V1_PREFIX)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application with routes and middleware."""
    _configure_logging()
    validate_security_configuration()
    docs_enabled = bool(settings.app.docs_enabled)
    app = FastAPI(
        title=settings.app.name,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"status": "ok"}

    _register_lifecycle_handlers(app)
    _add_enabled_middlewares(app)
    _include_api_routers(app)

    if FRONTEND_DIST_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="ui")
    return app


app = create_app()
