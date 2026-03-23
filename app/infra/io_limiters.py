import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.core.config import get_settings

settings = get_settings()

_LIMITS: dict[str, asyncio.Semaphore] = {
    "llm": asyncio.Semaphore(settings.io.llm_max_concurrency),
    "embedding": asyncio.Semaphore(settings.io.embedding_max_concurrency),
    "retrieval": asyncio.Semaphore(settings.io.retrieval_max_concurrency),
    "reranker": asyncio.Semaphore(settings.io.reranker_max_concurrency),
    "redis": asyncio.Semaphore(settings.io.redis_max_concurrency),
    "serpapi": asyncio.Semaphore(settings.serpapi.max_concurrency),
}


@asynccontextmanager
async def dependency_limiter(name: str) -> AsyncIterator[None]:
    """Acquire the configured semaphore for one downstream dependency."""
    semaphore = _LIMITS.get(name)
    if semaphore is None:
        raise ValueError(f"Unknown dependency limiter: {name}")
    async with semaphore:
        yield
