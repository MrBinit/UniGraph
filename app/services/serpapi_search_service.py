import asyncio
import json
import os
import urllib.parse
import urllib.request

from app.core.config import get_settings
from app.infra.io_limiters import dependency_limiter

settings = get_settings()


def _require_serpapi_enabled():
    if not settings.serpapi.enabled:
        raise RuntimeError("SerpAPI is disabled. Set serpapi.enabled=true in config.")


def _api_key() -> str:
    env_name = str(settings.serpapi.api_key_env_name).strip() or "SERPAPI_API_KEY"
    key = os.getenv(env_name, "").strip()
    if not key:
        raise RuntimeError(f"{env_name} is required for SerpAPI requests.")
    return key


def _search_url() -> str:
    url = str(settings.serpapi.google_search_url).strip()
    if not url:
        raise RuntimeError("serpapi.google_search_url must be configured.")
    return url


def _request_json(url: str, params: dict, timeout_seconds: float) -> dict:
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        request_url,
        headers={"User-Agent": "unigraph-serpapi-client/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("SerpAPI response must be a JSON object.")
    return payload


def _search_google_sync(
    query: str,
    *,
    gl: str | None = None,
    hl: str | None = None,
    num: int | None = None,
) -> dict:
    trimmed_query = str(query).strip()
    if not trimmed_query:
        raise ValueError("query must be non-empty.")

    params = {
        "engine": str(settings.serpapi.engine).strip() or "google",
        "q": trimmed_query,
        "gl": str(gl or settings.serpapi.default_gl).strip() or "us",
        "hl": str(hl or settings.serpapi.default_hl).strip() or "en",
        "num": max(1, int(num or settings.serpapi.default_num)),
        "api_key": _api_key(),
    }
    return _request_json(_search_url(), params, float(settings.serpapi.timeout_seconds))


def search_google(
    query: str,
    *,
    gl: str | None = None,
    hl: str | None = None,
    num: int | None = None,
) -> dict:
    """Run one SerpAPI Google Search request."""
    _require_serpapi_enabled()
    return _search_google_sync(query, gl=gl, hl=hl, num=num)


async def asearch_google(
    query: str,
    *,
    gl: str | None = None,
    hl: str | None = None,
    num: int | None = None,
) -> dict:
    """Run one SerpAPI Google Search request asynchronously."""
    _require_serpapi_enabled()
    async with dependency_limiter("serpapi"):
        return await asyncio.to_thread(_search_google_sync, query, gl=gl, hl=hl, num=num)


def _normalized_queries(queries: list[str]) -> list[str]:
    return [query.strip() for query in queries if isinstance(query, str) and query.strip()]


async def asearch_google_batch(
    queries: list[str],
    *,
    gl: str | None = None,
    hl: str | None = None,
    num: int | None = None,
) -> list[dict]:
    """Run many SerpAPI requests with an internal async work queue."""
    _require_serpapi_enabled()
    normalized = _normalized_queries(queries)
    if not normalized:
        return []

    results: list[dict] = [{"query": query, "result": {}, "error": ""} for query in normalized]
    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.serpapi.queue_max_size)
    worker_count = min(settings.serpapi.queue_workers, len(normalized))

    async def _worker():
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return

                index, query_value = item
                try:
                    payload = await asearch_google(query_value, gl=gl, hl=hl, num=num)
                    results[index] = {"query": query_value, "result": payload, "error": ""}
                except Exception as exc:
                    results[index] = {"query": query_value, "result": {}, "error": str(exc)}
            finally:
                queue.task_done()

    workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]

    for index, query_value in enumerate(normalized):
        await queue.put((index, query_value))
    for _ in range(worker_count):
        await queue.put(None)

    await queue.join()
    await asyncio.gather(*workers)
    return results
