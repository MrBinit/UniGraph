import os

import redis
from app.core.config import get_settings

settings = get_settings()


def _build_redis_client(config):
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
    return redis.Redis(**kwargs)


def _scoped_key(namespace: str, *parts: str) -> str:
    cleaned = [namespace.strip()]
    for part in parts:
        if part is None:
            continue
        text = str(part).strip(": ")
        if text:
            cleaned.append(text)
    return ":".join(cleaned)


def app_scoped_key(*parts: str) -> str:
    return _scoped_key(settings.redis.app.namespace, *parts)


def worker_scoped_key(*parts: str) -> str:
    return _scoped_key(settings.redis.worker.namespace, *parts)


app_redis_client = _build_redis_client(settings.redis.app)
worker_redis_client = _build_redis_client(settings.redis.worker)


def _runtime_role() -> str:
    role = os.getenv("REDIS_RUNTIME_ROLE", "app").strip().lower()
    return "worker" if role == "worker" else "app"


redis_client = worker_redis_client if _runtime_role() == "worker" else app_redis_client
