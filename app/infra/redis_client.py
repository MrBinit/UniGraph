import os
import ssl

import redis
import redis.asyncio as redis_async
from app.core.config import get_settings

settings = get_settings()


def _ssl_cert_reqs(cert_reqs: str):
    """Map app config values to ssl module verification constants."""
    value = (cert_reqs or "required").strip().lower()
    mapping = {
        "required": ssl.CERT_REQUIRED,
        "optional": ssl.CERT_OPTIONAL,
        "none": ssl.CERT_NONE,
    }
    return mapping.get(value, ssl.CERT_REQUIRED)


def _build_redis_client(config):
    """Create a Redis client from one role-specific Redis configuration block."""
    kwargs = {
        "host": config.host,
        "port": config.port,
        "db": config.db,
        "decode_responses": True,
    }
    if config.username:
        kwargs["username"] = config.username
    if config.password:
        kwargs["password"] = config.password
    if config.tls:
        kwargs["ssl"] = True
        kwargs["ssl_cert_reqs"] = _ssl_cert_reqs(config.ssl_cert_reqs)
        if config.ssl_ca_certs:
            kwargs["ssl_ca_certs"] = config.ssl_ca_certs
    return redis.Redis(**kwargs)


def _build_async_redis_client(config):
    """Create an async Redis client from one role-specific Redis config block."""
    kwargs = {
        "host": config.host,
        "port": config.port,
        "db": config.db,
        "decode_responses": True,
    }
    if config.username:
        kwargs["username"] = config.username
    if config.password:
        kwargs["password"] = config.password
    if config.tls:
        kwargs["ssl"] = True
        kwargs["ssl_cert_reqs"] = _ssl_cert_reqs(config.ssl_cert_reqs)
        if config.ssl_ca_certs:
            kwargs["ssl_ca_certs"] = config.ssl_ca_certs
    return redis_async.Redis(**kwargs)


def _scoped_key(namespace: str, *parts: str) -> str:
    """Join key parts into a normalized Redis key under the given namespace."""
    cleaned = [namespace.strip()]
    for part in parts:
        if part is None:
            continue
        text = str(part).strip(": ")
        if text:
            cleaned.append(text)
    return ":".join(cleaned)


def app_scoped_key(*parts: str) -> str:
    """Build a Redis key in the application namespace."""
    return _scoped_key(settings.redis.app.namespace, *parts)


def worker_scoped_key(*parts: str) -> str:
    """Build a Redis key in the worker namespace."""
    return _scoped_key(settings.redis.worker.namespace, *parts)


app_redis_client = _build_redis_client(settings.redis.app)
worker_redis_client = _build_redis_client(settings.redis.worker)
app_async_redis_client = _build_async_redis_client(settings.redis.app)
worker_async_redis_client = _build_async_redis_client(settings.redis.worker)


def _runtime_role() -> str:
    """Return the effective Redis runtime role used by the current process."""
    role = os.getenv("REDIS_RUNTIME_ROLE", "app").strip().lower()
    return "worker" if role == "worker" else "app"


redis_client = worker_redis_client if _runtime_role() == "worker" else app_redis_client
async_redis_client = (
    worker_async_redis_client if _runtime_role() == "worker" else app_async_redis_client
)
