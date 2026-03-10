import logging

from fastapi import FastAPI

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
from app.infra.postgres_client import get_postgres_pool
from app.infra.redis_client import app_redis_client

settings = get_settings()
logger = logging.getLogger(__name__)


def _configure_logging():
    """Initialize application logging from the configured log level."""
    level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


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

    @app.on_event("startup")
    async def warm_backends():
        """Warm backend connections during startup to reduce first-request latency."""
        try:
            app_redis_client.ping()
            logger.info("StartupWarmup | redis=ok")
        except Exception as exc:
            logger.warning("StartupWarmup | redis=failed | error=%s", exc)

        if settings.postgres.enabled:
            try:
                pool = get_postgres_pool()
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                logger.info("StartupWarmup | postgres=ok")
            except Exception as exc:
                logger.warning("StartupWarmup | postgres=failed | error=%s", exc)

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

    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(evaluation_router, prefix="/api/v1")
    app.include_router(ops_router, prefix="/api/v1")
    return app


app = create_app()
