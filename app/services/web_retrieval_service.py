import asyncio
import contextvars
import hashlib
import io
import inspect
import json
import logging
import re
import time
import urllib.request
from datetime import datetime, timezone
from html import unescape
from urllib.parse import urlparse

from app.core.config import get_settings
from app.infra.io_limiters import DependencyBackpressureError, dependency_limiter
from app.infra.redis_client import app_scoped_key, async_redis_client
from app.services.chat_trace_service import emit_trace_event
from app.services.tavily_search_service import asearch_google, asearch_google_batch
from redis.exceptions import RedisError

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency guard
    PdfReader = None

settings = get_settings()
logger = logging.getLogger(__name__)

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", flags=re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_BOILERPLATE_BLOCK_RE = re.compile(
    r"<(nav|footer|header|aside|form|noscript|svg)\b[^>]*>.*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)
_BLOCK_BREAK_RE = re.compile(
    r"</?(article|section|main|div|p|li|ul|ol|h[1-6]|table|tr|td|th|br)\b[^>]*>",
    flags=re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DEADLINE_HINT_RE = re.compile(
    r"\b(deadline|apply by|last date|closing date|intake)\b",
    flags=re.IGNORECASE,
)
_REQUIREMENTS_HINT_RE = re.compile(
    r"\b(requirements?|eligibility|admission|documents?)\b",
    flags=re.IGNORECASE,
)
_LANGUAGE_HINT_RE = re.compile(
    r"\b(language|ielts|toefl|english|german|international students?)\b",
    flags=re.IGNORECASE,
)
_CURRICULUM_HINT_RE = re.compile(
    r"\b(curriculum|module|course structure|syllabus)\b",
    flags=re.IGNORECASE,
)
_TUITION_HINT_RE = re.compile(
    r"\b(tuition|fees|semester contribution|cost)\b",
    flags=re.IGNORECASE,
)
_META_PUBLISHED_RE = [
    re.compile(
        r"<meta[^>]+(?:property|name)\s*=\s*[\"']"
        r"(?:article:published_time|publishdate|pubdate|date|dc\.date|og:updated_time)"
        r"[\"'][^>]+content\s*=\s*[\"']([^\"']+)[\"']",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"<meta[^>]+content\s*=\s*[\"']([^\"']+)[\"'][^>]+(?:property|name)\s*=\s*[\"']"
        r"(?:article:published_time|publishdate|pubdate|date|dc\.date|og:updated_time)"
        r"[\"']",
        flags=re.IGNORECASE,
    ),
]
_TIME_TAG_RE = re.compile(r"<time[^>]+datetime\s*=\s*[\"']([^\"']+)[\"']", flags=re.IGNORECASE)
_DATE_LIKE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    flags=re.IGNORECASE,
)
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_DEADLINE_CONTENT_RE = re.compile(
    r"\b(application deadline|deadline|application period|apply by|closing date|bewerbungsfrist|frist)\b",
    flags=re.IGNORECASE,
)
_DATE_VALUE_RE = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b|"
    r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
    r"sep|sept|september|oct|october|nov|november|dec|december)\b",
    flags=re.IGNORECASE,
)
_LANGUAGE_CONTENT_RE = re.compile(
    r"\b(language requirement|english proficiency|proof of english|ielts|toefl|cefr|german)\b",
    flags=re.IGNORECASE,
)
_LANGUAGE_SCORE_RE = re.compile(
    r"\b(ielts|toefl|cefr|unicert|cambridge)\b.{0,25}\b\d",
    flags=re.IGNORECASE,
)
_ADMISSION_CONTENT_RE = re.compile(
    r"\b(admission requirements?|eligibility|entry requirements?|bachelor|qualifying degree|documents?)\b",
    flags=re.IGNORECASE,
)
_DURATION_ECTS_CONTENT_RE = re.compile(
    r"\b(ects|credit points?|cp|semester|semesters|duration|years?)\b",
    flags=re.IGNORECASE,
)
_CURRICULUM_CONTENT_RE = re.compile(
    r"\b(curriculum|course structure|study plan|modules?|regulations?|pruefungsordnung)\b",
    flags=re.IGNORECASE,
)
_TUITION_CONTENT_RE = re.compile(
    r"\b(tuition|fees|semester contribution|costs?)\b",
    flags=re.IGNORECASE,
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}
_BOILERPLATE_LINE_MARKERS = {
    "cookie",
    "privacy policy",
    "terms of use",
    "all rights reserved",
    "sign in",
    "log in",
    "subscribe",
    "newsletter",
    "javascript is disabled",
    "enable javascript",
    "accept all",
}
_HIGH_AUTHORITY_SUFFIXES = (
    ".gov",
    ".edu",
    ".ac.uk",
    ".europa.eu",
)
_OFFICIAL_SOURCE_HOST_MARKERS = (
    "uni",
    "university",
    "universit",
    "universitaet",
    "hochschule",
    "college",
)
_OFFICIAL_SOURCE_HOST_PREFIXES = ("uni", "tu", "th", "fh", "hs")
_OFFICIAL_SOURCE_TEXT_MARKERS = (
    "university",
    "universität",
    "universitaet",
    "hochschule",
    "faculty",
    "department",
    "school of",
    "institute",
)
_ACADEMIC_PAGE_MARKERS = (
    "master",
    "m.sc",
    "msc",
    "admission",
    "requirements",
    "application",
    "deadline",
    "study program",
    "programme",
    "studium",
    "bewerbung",
)
_NON_OFFICIAL_HOST_MARKERS = (
    "blog",
    "forum",
    "wiki",
    "ranking",
    "rankings",
    "portal",
    "directory",
    "listing",
    "review",
    "consult",
    "newsroom",
)
_PLANNER_CACHE_VERSION = "v2"
_RETRIEVAL_MODE_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "web_retrieval_mode",
    default="deep",
)
_STANDARD_SEARCH_MODES = {"fast", "standard"}


def _normalized_search_mode(search_mode: str | None) -> str:
    candidate = str(search_mode or "").strip().lower()
    if candidate in {"deep", "fast", "standard"}:
        return candidate
    return "deep"


def _current_search_mode() -> str:
    return _normalized_search_mode(_RETRIEVAL_MODE_CTX.get())


def _is_deep_search_mode(search_mode: str | None = None) -> bool:
    mode = _normalized_search_mode(search_mode) if search_mode is not None else _current_search_mode()
    return mode not in _STANDARD_SEARCH_MODES


def _search_depth_for_mode(search_mode: str | None = None) -> str:
    return "advanced" if _is_deep_search_mode(search_mode) else "basic"


def _deep_mode_int_override(
    attr_name: str,
    configured: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if not _is_deep_search_mode():
        return configured
    override = getattr(settings.web_search, attr_name, configured)
    try:
        candidate = int(override)
    except (TypeError, ValueError):
        candidate = configured
    candidate = max(minimum, min(maximum, candidate))
    return max(configured, candidate)


def _default_num_for_mode(top_k: int) -> int:
    configured = max(1, int(settings.web_search.default_num))
    base = max(top_k, configured)
    return _deep_mode_int_override("deep_default_num", base, minimum=1, maximum=100)


def _max_query_variants_for_mode() -> int:
    configured = max(1, int(settings.web_search.max_query_variants))
    if _is_deep_search_mode():
        return _deep_mode_int_override("deep_max_query_variants", configured, minimum=1, maximum=8)
    # Standard/Fast mode stays lightweight for lower API credit burn.
    return min(2, configured)


def _max_context_results_for_mode() -> int:
    configured = max(1, int(settings.web_search.max_context_results))
    return _deep_mode_int_override("deep_max_context_results", configured, minimum=1, maximum=20)


def _max_pages_to_fetch_for_mode() -> int:
    configured = max(0, int(settings.web_search.max_pages_to_fetch))
    if configured <= 0:
        return 0
    return _deep_mode_int_override("deep_max_pages_to_fetch", configured, minimum=0, maximum=20)


def _max_chunks_per_page_for_mode() -> int:
    configured = max(1, int(settings.web_search.max_chunks_per_page))
    return _deep_mode_int_override("deep_max_chunks_per_page", configured, minimum=1, maximum=20)


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


async def _redis_call(method, *args, **kwargs):
    """Execute one Redis operation behind the shared Redis dependency limiter."""
    async with dependency_limiter("redis"):
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def _planner_cache_key(*, model_id: str, query: str, allowed_suffixes: list[str]) -> str:
    payload = {
        "version": _PLANNER_CACHE_VERSION,
        "model_id": model_id,
        "query": " ".join(str(query).split()).strip(),
        "allowed_suffixes": list(allowed_suffixes),
        "max_queries": _max_planner_queries(),
        "max_subquestions": _max_planner_subquestions(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return app_scoped_key("cache", "web_search", "query_planner", f"sha256:{digest}")


def _gap_planner_cache_key(
    *,
    model_id: str,
    query: str,
    subquestions: list[str],
    facts: list[dict],
    fallback_missing: list[str],
) -> str:
    compact_facts: list[dict[str, str]] = []
    for item in facts[:12]:
        if not isinstance(item, dict):
            continue
        compact_facts.append(
            {
                "fact": " ".join(str(item.get("fact", "")).split())[:180],
                "url": str(item.get("url", "")).strip()[:180],
            }
        )
    payload = {
        "version": _PLANNER_CACHE_VERSION,
        "model_id": model_id,
        "query": " ".join(str(query).split()).strip(),
        "subquestions": _normalize_subquestion_list(
            subquestions,
            limit=max(1, _max_planner_subquestions() or 1),
        ),
        "fallback_missing": _normalize_subquestion_list(
            fallback_missing,
            limit=max(1, _max_planner_subquestions() or 1),
        ),
        "facts": compact_facts,
        "max_gap_queries": max(
            1,
            int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2)),
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return app_scoped_key("cache", "web_search", "gap_planner", f"sha256:{digest}")


def _official_domains_for_query(query: str) -> list[str]:
    text = " ".join(str(query).split()).strip().lower()
    if not text:
        return []
    matches = re.findall(r"\b(?:[a-z0-9-]+\.)+(?:de|eu)\b", text)
    domains: list[str] = []
    seen: set[str] = set()
    for domain in matches:
        normalized = str(domain).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        domains.append(normalized)
    return domains


def _comparison_entities_from_query(query: str) -> list[str]:
    text = " ".join(str(query or "").split()).strip()
    if not text:
        return []
    match = re.search(
        r"(?:compare\s+)?(.+?)\s+(?:vs|versus)\s+(.+?)(?:,| for | including |$)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return []
    candidates = [match.group(1).strip(), match.group(2).strip()]
    entities: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        compact = " ".join(str(candidate).split()).strip()
        key = compact.lower()
        if not compact or key in seen:
            continue
        seen.add(key)
        entities.append(compact[:120])
    return entities[:2]


def _entity_focus_query(entity: str) -> str:
    compact_entity = " ".join(str(entity).split()).strip()
    if not compact_entity:
        return ""
    return (
        f"{compact_entity} data science master's program "
        "admission requirements application deadline"
    )


def _domain_group_key(host: str) -> str:
    normalized = str(host or "").strip().lower()
    if normalized.startswith("www."):
        normalized = normalized[4:]
    if not normalized:
        return ""
    parts = [segment for segment in normalized.split(".") if segment]
    if len(parts) <= 2:
        return normalized
    return ".".join(parts[-2:])


async def _read_cache_json(cache_key: str) -> dict | None:
    try:
        raw = await _redis_call(async_redis_client.get, cache_key)
    except RedisError as exc:
        logger.warning("Web-search planner cache read failed. %s", exc)
        return None
    if not raw:
        return None
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


async def _write_cache_json(cache_key: str, payload: dict, *, ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        return
    try:
        await _redis_call(
            async_redis_client.setex,
            cache_key,
            ttl_seconds,
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
        )
    except RedisError as exc:
        logger.warning("Web-search planner cache write failed. %s", exc)


def _extract_published_date(raw_html: str) -> str:
    source = raw_html[:200_000]
    for pattern in _META_PUBLISHED_RE:
        match = pattern.search(source)
        if not match:
            continue
        value = str(match.group(1) or "").strip()
        if value:
            return value[:80]
    match = _TIME_TAG_RE.search(source)
    if match:
        value = str(match.group(1) or "").strip()
        if value:
            return value[:80]
    return ""


def _is_boilerplate_line(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in _BOILERPLATE_LINE_MARKERS)


def _clean_html_text(raw_html: str, max_chars: int) -> str:
    text = _COMMENT_RE.sub(" ", raw_html)
    text = _SCRIPT_STYLE_RE.sub("\n", text)
    if settings.web_search.strip_boilerplate:
        text = _BOILERPLATE_BLOCK_RE.sub("\n", text)
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text).replace("\xa0", " ")
    min_line_chars = max(0, int(settings.web_search.min_clean_line_chars))

    lines: list[str] = []
    used_chars = 0
    for raw_line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line).strip(" |-\t")
        if not line:
            continue
        if min_line_chars and len(line) < min_line_chars:
            continue
        if settings.web_search.strip_boilerplate and _is_boilerplate_line(line):
            continue
        lines.append(line)
        used_chars += len(line) + 1
        if used_chars >= max_chars:
            break
    return "\n".join(lines)[:max_chars]


def _clean_plain_text(raw_text: str, max_chars: int) -> str:
    min_line_chars = max(0, int(settings.web_search.min_clean_line_chars))
    lines: list[str] = []
    used_chars = 0
    for raw_line in str(raw_text).splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line).strip(" |-\t")
        if not line:
            continue
        if min_line_chars and len(line) < min_line_chars:
            continue
        if settings.web_search.strip_boilerplate and _is_boilerplate_line(line):
            continue
        lines.append(line)
        used_chars += len(line) + 1
        if used_chars >= max_chars:
            break
    return "\n".join(lines)[:max_chars]


def _extract_pdf_text(raw_bytes: bytes, *, max_chars: int) -> str:
    if not raw_bytes or PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except Exception:
        return ""

    chunks: list[str] = []
    used_chars = 0
    for page in reader.pages:
        try:
            page_text = str(page.extract_text() or "").strip()
        except Exception:
            continue
        if not page_text:
            continue
        cleaned = _clean_plain_text(page_text, max_chars=max_chars)
        if not cleaned:
            continue
        chunks.append(cleaned)
        used_chars += len(cleaned) + 1
        if used_chars >= max_chars:
            break
    return "\n".join(chunks)[:max_chars]


def _fetch_page_data_sync(url: str, timeout_seconds: float, max_chars: int) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "unigraph-web-retrieval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        content_type = str(response.headers.get("Content-Type", "")).lower()
        max_bytes = max(4_000_000, max_chars * 24)
        raw_bytes = response.read(max_bytes)

    if "application/pdf" in content_type or str(url).lower().endswith(".pdf"):
        return {
            "content": _extract_pdf_text(raw_bytes, max_chars=max_chars),
            "published_date": "",
        }
    if "text/html" in content_type:
        raw = raw_bytes.decode("utf-8", errors="ignore")
        return {
            "content": _clean_html_text(raw, max_chars=max_chars),
            "published_date": _extract_published_date(raw),
        }
    if "text/" in content_type:
        raw = raw_bytes.decode("utf-8", errors="ignore")
        return {
            "content": _clean_plain_text(raw, max_chars=max_chars),
            "published_date": "",
        }

    return {
        "content": "",
        "published_date": "",
    }


def _row_published_date(row: dict) -> str:
    for key in ("date", "published_date", "published"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:80]
    extensions = row.get("extensions", [])
    if isinstance(extensions, list):
        for item in extensions:
            value = str(item).strip()
            if value and _DATE_LIKE_RE.search(value):
                return value[:80]
    return ""


def _organic_rows(payload: dict, limit: int) -> list[dict]:
    rows = payload.get("organic_results", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    results: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        link = str(row.get("link", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        if not (title or link or snippet):
            continue
        results.append(
            {
                "title": title,
                "url": link,
                "snippet": snippet,
                "published_date": _row_published_date(row),
            }
        )
        if len(results) >= limit:
            break
    return results


def _ai_overview_scalar_parts(ai: dict) -> list[str]:
    parts: list[str] = []
    for key in ("title", "text", "snippet", "description"):
        value = ai.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return parts


def _ai_overview_list_item_text(item) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    title = str(item.get("title", "")).strip()
    snippet = str(item.get("snippet", "")).strip()
    return f"{title}: {snippet}".strip(": ")


def _ai_overview_list_parts(items) -> list[str]:
    if not isinstance(items, list):
        return []
    parts: list[str] = []
    for item in items:
        text = _ai_overview_list_item_text(item)
        if text:
            parts.append(text)
    return parts


def _ai_overview_text(payload: dict) -> str:
    ai = payload.get("ai_overview", {}) if isinstance(payload, dict) else {}
    if not isinstance(ai, dict):
        return ""
    parts = _ai_overview_scalar_parts(ai)
    parts.extend(_ai_overview_list_parts(ai.get("list", [])))
    return " ".join(parts).strip()


async def _afetch_page_data(url: str) -> dict:
    async with dependency_limiter("web_search"):
        return await asyncio.to_thread(
            _fetch_page_data_sync,
            url,
            float(settings.web_search.page_fetch_timeout_seconds),
            int(settings.web_search.max_page_chars),
        )


async def _afetch_organic_pages(
    rows: list[dict], *, max_pages_to_fetch: int | None = None
) -> dict[str, dict]:
    if not settings.web_search.fetch_page_content:
        return {}

    targets = [row for row in rows if row.get("url")]
    page_limit = (
        max(0, int(max_pages_to_fetch))
        if max_pages_to_fetch is not None
        else _max_pages_to_fetch_for_mode()
    )
    targets = targets[:page_limit]
    if not targets:
        return {}

    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.web_search.queue_max_size)
    worker_count = min(settings.web_search.queue_workers, len(targets))
    fetched: dict[str, dict] = {}

    async def _worker():
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                url = str(item).strip()
                if not url:
                    continue
                try:
                    fetched[url] = await _afetch_page_data(url)
                except Exception:
                    fetched[url] = {"content": "", "published_date": ""}
            finally:
                queue.task_done()

    workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]
    for row in targets:
        await queue.put(row["url"])
    for _ in range(worker_count):
        await queue.put(None)

    await queue.join()
    await asyncio.gather(*workers)
    return fetched


def _host_label(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or "web"


def _compact_query_keywords(query: str) -> str:
    tokens = _QUERY_TOKEN_RE.findall(query.lower())
    if not tokens:
        return ""
    compact: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) <= 2 or token in _QUERY_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        compact.append(token)
        if len(compact) >= 8:
            break
    return " ".join(compact).strip()


def _normalized_allowed_domain_suffixes() -> list[str]:
    raw_values = getattr(settings.web_search, "allowed_domain_suffixes", [".de", ".eu"])
    if raw_values is None:
        raw_values = [".de", ".eu"]
    if isinstance(raw_values, str):
        values = [item.strip() for item in raw_values.split(",") if item.strip()]
    elif isinstance(raw_values, list):
        values = [str(item).strip() for item in raw_values if str(item).strip()]
    else:
        values = [".de", ".eu"]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        suffix = value.lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix in seen:
            continue
        seen.add(suffix)
        normalized.append(suffix)
    return normalized


def _normalized_official_source_allowlist() -> list[str]:
    raw_values = getattr(settings.web_search, "official_source_allowlist", [])
    if isinstance(raw_values, str):
        values = [item.strip().lower() for item in raw_values.split(",") if item.strip()]
    elif isinstance(raw_values, list):
        values = [str(item).strip().lower() for item in raw_values if str(item).strip()]
    else:
        values = []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        domain = value[4:] if value.startswith("www.") else value
        if domain in seen:
            continue
        seen.add(domain)
        normalized.append(domain)
    return normalized


def _host_matches_domain(host: str, domain: str) -> bool:
    normalized_host = _normalized_host(host)
    if not normalized_host:
        normalized_host = str(host or "").strip().lower()
        if normalized_host.startswith("www."):
            normalized_host = normalized_host[4:]
    normalized_domain = _normalized_host(domain)
    if not normalized_domain:
        normalized_domain = str(domain or "").strip().lower()
        if normalized_domain.startswith("www."):
            normalized_domain = normalized_domain[4:]
    if not normalized_host or not normalized_domain:
        return False
    return normalized_host == normalized_domain or normalized_host.endswith(f".{normalized_domain}")


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = str(text).lower()
    return any(marker in lowered for marker in markers)


def _host_looks_non_official(host: str) -> bool:
    grouped = _domain_group_key(host)
    if not grouped:
        return True
    host_text = grouped.replace("-", " ")
    return any(marker in host_text for marker in _NON_OFFICIAL_HOST_MARKERS)


def _host_looks_official_institution(host: str) -> bool:
    grouped = _domain_group_key(host)
    if not grouped:
        return False
    host_text = grouped.replace("-", " ")
    if any(marker in host_text for marker in _OFFICIAL_SOURCE_HOST_MARKERS):
        return True
    labels = [label for label in grouped.split(".") if label]
    return any(label.startswith(_OFFICIAL_SOURCE_HOST_PREFIXES) for label in labels)


def _source_url_allowed(
    *,
    url: str,
    title: str,
    snippet: str,
    allowed_suffixes: list[str],
) -> bool:
    if not _url_matches_allowed_suffix(url, allowed_suffixes):
        return False
    if not bool(getattr(settings.web_search, "official_source_filter_enabled", True)):
        return True
    host = _normalized_host(url)
    if not host:
        return False

    allowlist = _normalized_official_source_allowlist()
    if any(_host_matches_domain(host, domain) for domain in allowlist):
        return True

    if _host_looks_non_official(host):
        return False
    if _host_looks_official_institution(host):
        return True

    evidence_text = f"{title} {snippet}"
    if not _contains_marker(evidence_text, _OFFICIAL_SOURCE_TEXT_MARKERS):
        return False
    return _contains_marker(evidence_text, _ACADEMIC_PAGE_MARKERS)


def _build_query_variants(query: str, allowed_suffixes: list[str]) -> list[str]:
    base = " ".join(str(query).split()).strip()
    if not base:
        return []

    if not settings.web_search.multi_query_enabled:
        return [base]

    if not _is_deep_search_mode():
        official_domains = _official_domains_for_query(base)[:1]
        candidates = [base]
        if official_domains:
            candidates.append(f"{base} site:{official_domains[0]}")
        else:
            candidates.append(f"{base} official information")

        max_variants = _max_query_variants_for_mode()
        variants: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = " ".join(str(candidate).split()).strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            variants.append(normalized)
            if len(variants) >= max_variants:
                break
        return variants or [base]

    official_domains = _official_domains_for_query(base)[:3]
    comparison_entities = _comparison_entities_from_query(base)
    candidates: list[str] = [base]
    for entity in comparison_entities:
        entity_focus = _entity_focus_query(entity)
        if not entity_focus:
            continue
        candidates.append(entity_focus)
        for domain in _official_domains_for_query(entity)[:1]:
            candidates.append(f"{entity_focus} site:{domain}")
    for domain in official_domains:
        candidates.append(f"{base} site:{domain}")
    candidates.append(f"{base} official information")
    compact = _compact_query_keywords(base)
    if compact and compact != base.lower():
        candidates.append(compact)

    if allowed_suffixes:
        site_terms = " OR ".join(f"site:{suffix}" for suffix in allowed_suffixes[:2])
        candidates.append(f"{base} ({site_terms})")
    if _DEADLINE_HINT_RE.search(base):
        candidates.append(f"{base} application deadline official")
    if _REQUIREMENTS_HINT_RE.search(base):
        candidates.append(f"{base} admission requirements official")

    for domain in official_domains:
        if compact:
            candidates.append(f"{compact} site:{domain}")

    max_variants = _max_query_variants_for_mode()
    variants: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(str(candidate).split()).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        variants.append(normalized)
        if len(variants) >= max_variants:
            break
    return variants or [base]


def _normalize_query_list(values, *, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = " ".join(str(value).split()).strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)
        if len(normalized) >= limit:
            break
    return normalized


def _normalize_subquestion_list(values, *, limit: int) -> list[str]:
    return _normalize_query_list(values, limit=limit)


def _planner_model_id() -> str:
    configured = str(getattr(settings.web_search, "query_planner_model_id", "")).strip()
    return configured or str(settings.bedrock.primary_model_id).strip()


def _max_planner_queries() -> int:
    return max(1, int(getattr(settings.web_search, "query_planner_max_queries", 5)))


def _max_planner_subquestions() -> int:
    return max(0, int(getattr(settings.web_search, "query_planner_max_subquestions", 4)))


def _required_fields_from_query(query: str) -> list[dict]:
    compact = " ".join(str(query or "").split()).strip().lower()
    if not compact:
        return []
    field_catalog = {
        "program_overview": {
            "id": "program_overview",
            "label": "program overview",
            "subquestion": "program overview, degree type, department, and teaching language",
            "query_focus": "official program overview degree language",
        },
        "admission_requirements": {
            "id": "admission_requirements",
            "label": "course requirements",
            "subquestion": "course requirements, eligibility criteria, and required documents",
            "query_focus": "admission requirements eligibility required documents",
        },
        "language_requirements": {
            "id": "language_requirements",
            "label": "language requirements",
            "subquestion": "language requirements with accepted tests and minimum scores",
            "query_focus": "language requirements IELTS TOEFL minimum score",
        },
        "application_deadline": {
            "id": "application_deadline",
            "label": "application deadlines",
            "subquestion": "application deadline and intake timeline with exact dates",
            "query_focus": "application deadline exact dates intake timeline",
        },
        "duration_ects": {
            "id": "duration_ects",
            "label": "duration and ECTS",
            "subquestion": "program duration in semesters/years and total ECTS credits",
            "query_focus": "program duration semesters years total ECTS credits",
        },
        "curriculum_modules": {
            "id": "curriculum_modules",
            "label": "curriculum and modules",
            "subquestion": "curriculum structure and core modules from official regulations",
            "query_focus": "curriculum structure core modules regulations",
        },
        "tuition_fees": {
            "id": "tuition_fees",
            "label": "tuition and fees",
            "subquestion": "tuition fees and semester contribution amounts",
            "query_focus": "tuition fees semester contribution costs",
        },
    }
    selected_ids: list[str] = []
    explicit_admission_scope = bool(
        re.search(r"\b(admission|eligibility|entry|documents?|course requirements?)\b", compact)
    )
    language_only_requirements = bool(_LANGUAGE_HINT_RE.search(compact)) and not explicit_admission_scope
    if _REQUIREMENTS_HINT_RE.search(compact) and not language_only_requirements:
        selected_ids.append("admission_requirements")
    if _LANGUAGE_HINT_RE.search(compact):
        selected_ids.append("language_requirements")
    if _DEADLINE_HINT_RE.search(compact) or "application" in compact:
        selected_ids.append("application_deadline")
    if _CURRICULUM_HINT_RE.search(compact) or "course" in compact:
        selected_ids.append("curriculum_modules")
    if _TUITION_HINT_RE.search(compact):
        selected_ids.append("tuition_fees")
    if re.search(r"\b(duration|ects|credit|semester|year)\b", compact):
        selected_ids.append("duration_ects")

    if re.search(r"\b(master|m\.sc|msc|program|course|study)\b", compact):
        selected_ids.insert(0, "program_overview")

    if not selected_ids:
        if re.search(r"\b(master|m\.sc|msc|program|course|degree)\b", compact):
            selected_ids = ["program_overview"]
        else:
            selected_ids = []

    normalized: list[dict] = []
    seen: set[str] = set()
    for field_id in selected_ids:
        if field_id in seen:
            continue
        seen.add(field_id)
        field = field_catalog.get(field_id)
        if field:
            normalized.append(dict(field))
    return normalized


def _required_field_subquestions(required_fields: list[dict]) -> list[str]:
    items = [str(field.get("subquestion", "")).strip() for field in required_fields]
    return [item for item in items if item]


def _coverage_subquestions_from_query(query: str) -> list[str]:
    candidates: list[str] = []
    required_fields = _required_fields_from_query(query)
    candidates.extend(_required_field_subquestions(required_fields))
    compact = " ".join(str(query or "").split()).strip().lower()
    if not compact:
        return _normalize_subquestion_list(candidates, limit=_max_planner_subquestions())
    explicit_admission_scope = bool(
        re.search(r"\b(admission|eligibility|entry|documents?|course requirements?)\b", compact)
    )
    language_only_requirements = bool(_LANGUAGE_HINT_RE.search(compact)) and not explicit_admission_scope
    if _REQUIREMENTS_HINT_RE.search(compact) and not language_only_requirements:
        candidates.append("course requirements and eligibility criteria")
    if _LANGUAGE_HINT_RE.search(compact):
        candidates.append("language requirements and accepted English test minimum scores with exact numbers")
    if _DEADLINE_HINT_RE.search(compact):
        candidates.append("application deadline and intake timeline with exact dates")
    if _CURRICULUM_HINT_RE.search(compact):
        candidates.append("curriculum structure and core modules")
    if _TUITION_HINT_RE.search(compact):
        candidates.append("tuition and semester fees")
    return _normalize_subquestion_list(candidates, limit=_max_planner_subquestions())


def _build_query_planner_messages(query: str, allowed_suffixes: list[str]) -> list[dict]:
    max_queries = _max_planner_queries()
    max_subquestions = _max_planner_subquestions()
    suffix_clause = ", ".join(allowed_suffixes[:3]) if allowed_suffixes else "none"
    required_fields = _required_fields_from_query(query)
    focus_subquestions = _coverage_subquestions_from_query(query)
    focus_text = "\n".join(f"- {item}" for item in focus_subquestions) or "- (none)"
    required_field_text = (
        "\n".join(
            f"- {field.get('label', '')}: {field.get('query_focus', '')}" for field in required_fields
        )
        or "- (none)"
    )
    system_prompt = (
        "You are a web search query planner for high-coverage deep retrieval. "
        "Think in phases: plan -> search fan-out -> evidence fan-in. "
        "Return strict JSON only with keys: queries, subquestions. "
        "queries must preserve key entities, numbers, dates, and negations from the user query. "
        "Make queries independent and non-overlapping so they can run in parallel."
    )
    user_prompt = (
        f"User query: {query}\n"
        f"Allowed domain suffixes (optional): {suffix_clause}\n"
        f"Coverage dimensions to include when relevant:\n{focus_text}\n"
        f"Required field-focused query intents:\n{required_field_text}\n"
        "For fields asking deadlines/scores/duration, include dedicated exact-number/date queries.\n"
        "Prioritize official university pages and DAAD pages.\n"
        f"Return at most {max_queries} queries and at most {max_subquestions} subquestions."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json_object(raw: str) -> dict | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _heuristic_subquestions(query: str) -> list[str]:
    compact = " ".join(str(query).split()).strip()
    if not compact:
        return []
    parts = re.split(r"\s+(?:and|vs|versus)\s+", compact, flags=re.IGNORECASE)
    normalized = _normalize_subquestion_list(parts, limit=_max_planner_subquestions())
    return normalized if len(normalized) > 1 else []


def _build_heuristic_query_plan(query: str, allowed_suffixes: list[str]) -> dict:
    return {
        "queries": _build_query_variants(query, allowed_suffixes),
        "subquestions": _heuristic_subquestions(query),
        "planner": "heuristic",
        "llm_used": False,
    }


def _normalize_query_plan_payload(
    *,
    query: str,
    allowed_suffixes: list[str],
    payload: dict,
) -> dict:
    max_queries = _max_planner_queries()
    max_subquestions = _max_planner_subquestions()
    base_queries = [query] + _build_query_variants(query, allowed_suffixes)
    llm_queries = _normalize_query_list(payload.get("queries"), limit=max_queries)
    focus_queries = _build_gap_queries(query, _coverage_subquestions_from_query(query))
    merged_queries = _normalize_query_list(
        base_queries + llm_queries + focus_queries,
        limit=max_queries,
    )
    merged_subquestions = _normalize_subquestion_list(
        _coverage_subquestions_from_query(query)
        + _normalize_subquestion_list(
            payload.get("subquestions"),
            limit=max_subquestions,
        ),
        limit=max_subquestions,
    )
    return {
        "queries": merged_queries or _build_query_variants(query, allowed_suffixes),
        "subquestions": merged_subquestions,
        "planner": "llm",
        "llm_used": True,
    }


async def _aplan_queries_with_llm(query: str, allowed_suffixes: list[str]) -> dict | None:
    if not bool(getattr(settings.web_search, "query_planner_use_llm", False)):
        return None

    model_id = _planner_model_id()
    if not model_id:
        return None

    cache_key = _planner_cache_key(model_id=model_id, query=query, allowed_suffixes=allowed_suffixes)
    if bool(getattr(settings.web_search, "query_planner_cache_enabled", True)):
        cached = await _read_cache_json(cache_key)
        if cached:
            cached_plan = _normalize_query_plan_payload(
                query=query,
                allowed_suffixes=allowed_suffixes,
                payload=cached,
            )
            cached_plan["planner"] = "llm_cache"
            cached_plan["llm_used"] = True
            emit_trace_event(
                "query_plan_cache_hit",
                {
                    "query": query[:220],
                    "model_id": model_id,
                },
            )
            return cached_plan

    try:
        messages = _build_query_planner_messages(query, allowed_suffixes)
        from app.infra.bedrock_chat_client import client

        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            limiter_name="llm_planner",
            limiter_acquire_timeout_seconds=float(
                getattr(settings.web_search, "query_planner_acquire_timeout_seconds", 0.0)
            ),
            rate_limit_profile="planner",
        )
    except DependencyBackpressureError as exc:
        logger.warning(
            "Web-search query planner backpressure; falling back to heuristic variants. %s",
            exc,
        )
        emit_trace_event(
            "query_plan_backpressure",
            {
                "query": query[:220],
                "retry_after_seconds": round(float(exc.retry_after_seconds), 3),
            },
        )
        return None
    except Exception as exc:
        logger.warning(
            "Web-search query planner failed; falling back to heuristic variants. %s",
            exc,
        )
        return None

    if not response or not getattr(response, "choices", None):
        return None
    content = str(response.choices[0].message.content or "").strip()
    payload = _extract_json_object(content)
    if not payload:
        return None
    plan = _normalize_query_plan_payload(
        query=query,
        allowed_suffixes=allowed_suffixes,
        payload=payload,
    )
    if bool(getattr(settings.web_search, "query_planner_cache_enabled", True)):
        await _write_cache_json(
            cache_key,
            {
                "queries": plan.get("queries", []),
                "subquestions": plan.get("subquestions", []),
            },
            ttl_seconds=int(getattr(settings.web_search, "query_planner_cache_ttl_seconds", 900)),
        )
    return plan


def _planner_enabled() -> bool:
    return _is_deep_search_mode() and bool(getattr(settings.web_search, "query_planner_enabled", True))


async def _resolve_query_plan(query: str, allowed_suffixes: list[str]) -> dict:
    if not _planner_enabled():
        return _build_heuristic_query_plan(query, allowed_suffixes)
    llm_plan = await _aplan_queries_with_llm(query, allowed_suffixes)
    if llm_plan:
        return llm_plan
    return _build_heuristic_query_plan(query, allowed_suffixes)


def _retrieval_loop_model_id() -> str:
    configured = str(getattr(settings.web_search, "retrieval_loop_model_id", "")).strip()
    return configured or _planner_model_id()


def _loop_llm_enabled() -> bool:
    return _is_deep_search_mode() and bool(getattr(settings.web_search, "retrieval_loop_use_llm", False))


def _compact_facts_for_prompt(facts: list[dict], *, limit: int) -> list[str]:
    lines: list[str] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        fact = " ".join(str(item.get("fact", "")).split()).strip()
        if not fact:
            continue
        url = str(item.get("url", "")).strip()
        if url:
            lines.append(f"- {fact} | {url}")
        else:
            lines.append(f"- {fact}")
        if len(lines) >= limit:
            break
    return lines


def _build_gap_analyzer_messages(
    query: str,
    *,
    subquestions: list[str],
    facts: list[dict],
) -> list[dict]:
    max_gap_queries = max(1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2)))
    compact_facts = _compact_facts_for_prompt(facts, limit=12)
    subquestion_text = "\n".join(f"- {item}" for item in subquestions) or "- (none)"
    facts_text = "\n".join(compact_facts) or "- (none)"
    system_prompt = (
        "You analyze retrieval coverage. "
        "Reason silently and return JSON only with keys: missing_subquestions, queries. "
        "Do not include explanations."
    )
    user_prompt = (
        f"User query: {query}\n"
        f"Subquestions:\n{subquestion_text}\n"
        f"Extracted facts:\n{facts_text}\n"
        f"Return at most {max_gap_queries} targeted follow-up queries."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _normalize_gap_plan_payload(payload: dict, *, query: str, fallback_missing: list[str]) -> dict:
    max_gap_queries = max(1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2)))
    missing = _normalize_subquestion_list(
        payload.get("missing_subquestions"), limit=_max_planner_subquestions()
    )
    if not missing:
        missing = list(fallback_missing)
    queries = _normalize_query_list(payload.get("queries"), limit=max_gap_queries)
    if not queries and missing:
        queries = _build_gap_queries(query, missing)
    return {
        "missing_subquestions": missing,
        "queries": queries,
    }


async def _aidentify_gap_plan_with_llm(
    query: str,
    *,
    subquestions: list[str],
    facts: list[dict],
    fallback_missing: list[str],
) -> dict | None:
    if not _loop_llm_enabled() or not subquestions:
        return None
    model_id = _retrieval_loop_model_id()
    if not model_id:
        return None

    cache_key = _gap_planner_cache_key(
        model_id=model_id,
        query=query,
        subquestions=subquestions,
        facts=facts,
        fallback_missing=fallback_missing,
    )
    if bool(getattr(settings.web_search, "retrieval_loop_cache_enabled", True)):
        cached = await _read_cache_json(cache_key)
        if cached:
            plan = _normalize_gap_plan_payload(
                cached,
                query=query,
                fallback_missing=fallback_missing,
            )
            emit_trace_event(
                "retrieval_gap_plan_cache_hit",
                {
                    "query": query[:220],
                    "model_id": model_id,
                },
            )
            return plan

    try:
        messages = _build_gap_analyzer_messages(
            query,
            subquestions=subquestions,
            facts=facts,
        )
        from app.infra.bedrock_chat_client import client

        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            limiter_name="llm_planner",
            limiter_acquire_timeout_seconds=float(
                getattr(settings.web_search, "retrieval_loop_acquire_timeout_seconds", 0.0)
            ),
            rate_limit_profile="planner",
        )
    except DependencyBackpressureError as exc:
        logger.warning(
            "Web-search retrieval-loop backpressure; falling back to heuristic gap detection. %s",
            exc,
        )
        emit_trace_event(
            "retrieval_gap_plan_backpressure",
            {
                "query": query[:220],
                "retry_after_seconds": round(float(exc.retry_after_seconds), 3),
            },
        )
        return None
    except Exception as exc:
        logger.warning(
            "Web-search retrieval-loop gap analysis failed; "
            "falling back to heuristic gap detection. %s",
            exc,
        )
        return None

    if not response or not getattr(response, "choices", None):
        return None
    content = str(response.choices[0].message.content or "").strip()
    payload = _extract_json_object(content)
    if not payload:
        return None
    plan = _normalize_gap_plan_payload(
        payload,
        query=query,
        fallback_missing=fallback_missing,
    )
    if bool(getattr(settings.web_search, "retrieval_loop_cache_enabled", True)):
        await _write_cache_json(
            cache_key,
            {
                "missing_subquestions": plan.get("missing_subquestions", []),
                "queries": plan.get("queries", []),
            },
            ttl_seconds=int(getattr(settings.web_search, "retrieval_loop_cache_ttl_seconds", 300)),
        )
    return plan


def _subquestion_token_coverage(subquestion: str, evidence_text: str) -> float:
    tokens = _token_signature(subquestion)
    if not tokens:
        return 1.0
    evidence_tokens = _token_signature(evidence_text)
    if not evidence_tokens:
        return 0.0
    return len(tokens & evidence_tokens) / max(1, len(tokens))


def _identify_missing_subquestions(subquestions: list[str], facts: list[dict]) -> list[str]:
    if not subquestions:
        return []
    evidence_text = " ".join(str(item.get("fact", "")) for item in facts if isinstance(item, dict))
    threshold = float(getattr(settings.web_search, "retrieval_gap_min_token_coverage", 0.5))
    missing: list[str] = []
    for subquestion in subquestions:
        if _subquestion_token_coverage(subquestion, evidence_text) >= threshold:
            continue
        missing.append(subquestion)
    return missing


def _candidate_evidence_text(candidate: dict) -> str:
    if not isinstance(candidate, dict):
        return ""
    metadata = candidate.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    parts = [
        str(candidate.get("content", "")),
        str(metadata.get("title", "")),
        str(metadata.get("snippet", "")),
    ]
    return " ".join(part for part in parts if part).strip().lower()


def _required_field_covered_by_text(field_id: str, text: str) -> bool:
    compact = " ".join(str(text).split()).strip().lower()
    if not compact:
        return False
    if field_id == "program_overview":
        return bool(
            re.search(r"\b(master|m\.sc|msc|program|programme|department|english)\b", compact)
        )
    if field_id == "admission_requirements":
        return bool(_ADMISSION_CONTENT_RE.search(compact))
    if field_id == "language_requirements":
        return bool(
            _LANGUAGE_CONTENT_RE.search(compact)
            and (_LANGUAGE_SCORE_RE.search(compact) or _NUMERIC_TOKEN_RE.search(compact))
        )
    if field_id == "application_deadline":
        return bool(_DEADLINE_CONTENT_RE.search(compact) and _DATE_VALUE_RE.search(compact))
    if field_id == "duration_ects":
        return bool(_DURATION_ECTS_CONTENT_RE.search(compact) and _NUMERIC_TOKEN_RE.search(compact))
    if field_id == "curriculum_modules":
        return bool(_CURRICULUM_CONTENT_RE.search(compact))
    if field_id == "tuition_fees":
        return bool(_TUITION_CONTENT_RE.search(compact) and _NUMERIC_TOKEN_RE.search(compact))
    return False


def _required_field_coverage(
    required_fields: list[dict],
    candidates: list[dict],
) -> dict:
    if not required_fields:
        return {
            "fields": [],
            "missing_ids": [],
            "missing_labels": [],
            "missing_subquestions": [],
            "coverage": 1.0,
        }

    evidence_texts = [_candidate_evidence_text(item) for item in candidates if isinstance(item, dict)]
    statuses: list[dict] = []
    missing_ids: list[str] = []
    missing_labels: list[str] = []
    missing_subquestions: list[str] = []

    for field in required_fields:
        field_id = str(field.get("id", "")).strip()
        if not field_id:
            continue
        covered = any(_required_field_covered_by_text(field_id, text) for text in evidence_texts)
        label = str(field.get("label", field_id)).strip() or field_id
        statuses.append({"id": field_id, "label": label, "covered": covered})
        if covered:
            continue
        missing_ids.append(field_id)
        missing_labels.append(label)
        subquestion = " ".join(str(field.get("subquestion", "")).split()).strip()
        if subquestion:
            missing_subquestions.append(subquestion)

    total = len(statuses)
    covered_count = len([item for item in statuses if item.get("covered")])
    coverage = 1.0 if total <= 0 else (covered_count / total)
    return {
        "fields": statuses,
        "missing_ids": missing_ids,
        "missing_labels": missing_labels,
        "missing_subquestions": _normalize_subquestion_list(
            missing_subquestions,
            limit=max(1, _max_planner_subquestions()),
        ),
        "coverage": max(0.0, min(1.0, coverage)),
    }


def _required_fields_by_ids(required_fields: list[dict], missing_ids: list[str]) -> list[dict]:
    if not required_fields or not missing_ids:
        return []
    missing_set = {str(item).strip() for item in missing_ids if str(item).strip()}
    return [
        field
        for field in required_fields
        if str(field.get("id", "")).strip() in missing_set
    ]


def _build_gap_queries(query: str, missing_subquestions: list[str]) -> list[str]:
    if not missing_subquestions:
        return []
    max_gap_queries = max(1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2)))
    candidates = [
        f"{query} {subquestion}".strip() for subquestion in missing_subquestions[:max_gap_queries]
    ]
    return _normalize_query_list(candidates, limit=max_gap_queries)


def _next_queries_for_loop(
    planned_queries: list[str],
    seen_queries: set[str],
    *,
    max_queries: int | None = None,
) -> list[str]:
    query_limit = max_queries if isinstance(max_queries, int) and max_queries > 0 else _max_planner_queries()
    next_queries: list[str] = []
    for query in planned_queries:
        key = str(query).strip().lower()
        if not key or key in seen_queries:
            continue
        seen_queries.add(key)
        next_queries.append(str(query).strip())
        if len(next_queries) >= query_limit:
            break
    return next_queries


def _url_matches_allowed_suffix(url: str, allowed_suffixes: list[str]) -> bool:
    if not allowed_suffixes:
        return True
    host = str(urlparse(url).hostname or "").strip().lower()
    if not host:
        return False
    return any(host.endswith(suffix) for suffix in allowed_suffixes)


def _filter_rows_by_allowed_domains(rows: list[dict], allowed_suffixes: list[str]) -> list[dict]:
    filtered: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url", "")).strip()
        title = str(row.get("title", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        if not _source_url_allowed(
            url=url,
            title=title,
            snippet=snippet,
            allowed_suffixes=allowed_suffixes,
        ):
            continue
        filtered.append(row)
    return filtered


def _dedupe_rows(rows: list[dict], limit: int) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url", "")).strip()
        if url:
            key = f"url:{url.lower()}"
        else:
            title = str(row.get("title", "")).strip().lower()
            snippet = str(row.get("snippet", "")).strip().lower()
            key = f"text:{' '.join(f'{title} {snippet}'.split())[:220]}"
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def _wrap_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    parts: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current)
            current = word
    parts.append(current)
    return parts


def _segment_text_for_chunking(text: str, max_chars: int) -> list[str]:
    segments: list[str] = []
    for line in text.splitlines():
        normalized = _WHITESPACE_RE.sub(" ", line).strip()
        if not normalized:
            continue
        sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(normalized) if item.strip()]
        if not sentences:
            sentences = [normalized]
        for sentence in sentences:
            if len(sentence) <= max_chars:
                segments.append(sentence)
            else:
                segments.extend(_wrap_words(sentence, max_chars))
    return segments


def _append_chunk_if_ready(
    chunks: list[str],
    current: str,
    *,
    min_chunk_chars: int,
    max_chunks: int,
) -> bool:
    if len(current) < min_chunk_chars:
        return False
    chunks.append(current)
    return len(chunks) >= max_chunks


def _next_current_segment(chunks: list[str], *, overlap: int, segment: str) -> str:
    if overlap > 0 and chunks:
        tail = chunks[-1][-overlap:].strip()
        return f"{tail} {segment}".strip()
    return segment


def _finalize_chunks(
    *,
    chunks: list[str],
    current: str,
    segments: list[str],
    max_chars: int,
    min_chunk_chars: int,
    max_chunks: int,
) -> list[str]:
    if current and len(chunks) < max_chunks:
        if len(current) >= min_chunk_chars or not chunks:
            chunks.append(current)
    if not chunks and segments:
        chunks.append(segments[0][:max_chars])
    return chunks[:max_chunks]


def _chunk_clean_text(clean_text: str) -> list[str]:
    max_chars = max(120, int(settings.web_search.page_chunk_chars))
    overlap = max(0, min(int(settings.web_search.page_chunk_overlap_chars), max_chars // 2))
    max_chunks = _max_chunks_per_page_for_mode()
    min_chunk_chars = max(20, int(settings.web_search.min_chunk_chars))
    segments = _segment_text_for_chunking(clean_text, max_chars=max_chars)
    if not segments:
        return []

    chunks: list[str] = []
    current = ""
    for segment in segments:
        if not current:
            current = segment
            continue
        candidate = f"{current} {segment}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if _append_chunk_if_ready(
            chunks,
            current,
            min_chunk_chars=min_chunk_chars,
            max_chunks=max_chunks,
        ):
            return chunks
        current = _next_current_segment(chunks, overlap=overlap, segment=segment)

    return _finalize_chunks(
        chunks=chunks,
        current=current,
        segments=segments,
        max_chars=max_chars,
        min_chunk_chars=min_chunk_chars,
        max_chunks=max_chunks,
    )


def _token_signature(text: str) -> set[str]:
    return {
        token
        for token in _QUERY_TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _QUERY_STOPWORDS
    }


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union <= 0:
        return 0.0
    return intersection / union


def _dedupe_chunk_candidates(candidates: list[dict]) -> list[dict]:
    threshold = float(settings.web_search.chunk_dedupe_similarity)
    deduped: list[dict] = []
    signatures: list[tuple[set[str], str]] = []
    for candidate in candidates:
        extracted = _candidate_signature_and_url(candidate)
        if not extracted:
            continue
        signature, url = extracted
        if _is_duplicate_candidate(signature, url, signatures, threshold):
            continue
        deduped.append(candidate)
        signatures.append((signature, url))
    return deduped


def _candidate_signature_and_url(candidate: dict) -> tuple[set[str], str] | None:
    content = str(candidate.get("content", "")).strip()
    if not content:
        return None
    signature = _token_signature(content)
    if not signature:
        signature = {content.lower()[:80]}
    metadata = candidate.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    url = str(metadata.get("url", "")).strip().lower()
    return signature, url


def _is_duplicate_candidate(
    signature: set[str],
    url: str,
    prior_signatures: list[tuple[set[str], str]],
    threshold: float,
) -> bool:
    for prior_signature, prior_url in prior_signatures:
        similarity = _jaccard_similarity(signature, prior_signature)
        if similarity < threshold:
            continue
        same_url = bool(url and prior_url and url == prior_url)
        rich_enough = min(len(signature), len(prior_signature)) >= 6
        if same_url or rich_enough:
            return True
    return False


def _query_tokens(query: str) -> set[str]:
    return _token_signature(query)


def _chunk_relevance_score(
    *,
    query_tokens: set[str],
    content: str,
    snippet: str,
    rank_index: int,
) -> float:
    if not content:
        return 0.0
    content_tokens = _token_signature(content)
    overlap = 0.0
    if query_tokens and content_tokens:
        overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
    snippet_bonus = 0.0
    if snippet and snippet.lower() in content.lower():
        snippet_bonus = 0.05
    rank_bonus = max(0.0, 0.08 - ((rank_index - 1) * 0.01))
    return overlap + snippet_bonus + rank_bonus


def _domain_authority_score(url: str, allowed_suffixes: list[str]) -> float:
    host = str(urlparse(url).hostname or "").strip().lower()
    if not host:
        return 0.35
    if allowed_suffixes and any(host.endswith(suffix) for suffix in allowed_suffixes):
        return 0.82
    if any(host.endswith(suffix) for suffix in _HIGH_AUTHORITY_SUFFIXES):
        return 0.95
    if host.endswith(".org"):
        return 0.7
    if host.endswith(".com") or host.endswith(".net"):
        return 0.55
    return 0.5


def _parse_published_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    for candidate in (normalized, normalized[:10]):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _recency_score(published_date: str) -> float:
    parsed = _parse_published_datetime(published_date)
    if parsed is None:
        return 0.5
    now = datetime.now(timezone.utc)
    age_days = max(0, int((now - parsed).total_seconds() // 86400))
    if age_days <= 30:
        return 1.0
    if age_days <= 180:
        return 0.85
    if age_days <= 365:
        return 0.7
    if age_days <= 730:
        return 0.55
    return 0.4


def _agreement_score(candidate: dict, candidates: list[dict]) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    extracted = _candidate_signature_and_url(candidate)
    if not extracted:
        return 0.0
    signature, url = extracted
    support = 0
    for peer in candidates:
        if peer is candidate:
            continue
        peer_extracted = _candidate_signature_and_url(peer)
        if not peer_extracted:
            continue
        peer_signature, peer_url = peer_extracted
        if url and peer_url and url == peer_url:
            continue
        if _jaccard_similarity(signature, peer_signature) >= 0.2:
            support += 1
            if support >= 3:
                break
    return min(1.0, support / 3.0)


def _normalized_trust_weights() -> tuple[float, float, float, float]:
    relevance = max(0.0, float(getattr(settings.web_search, "trust_relevance_weight", 0.6)))
    authority = max(0.0, float(getattr(settings.web_search, "trust_authority_weight", 0.2)))
    recency = max(0.0, float(getattr(settings.web_search, "trust_recency_weight", 0.1)))
    agreement = max(0.0, float(getattr(settings.web_search, "trust_agreement_weight", 0.1)))
    total = relevance + authority + recency + agreement
    if total <= 0:
        return 0.6, 0.2, 0.1, 0.1
    return (
        relevance / total,
        authority / total,
        recency / total,
        agreement / total,
    )


def _apply_trust_scores(candidates: list[dict], allowed_suffixes: list[str]) -> list[dict]:
    relevance_w, authority_w, recency_w, agreement_w = _normalized_trust_weights()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            candidate["metadata"] = metadata
        url = str(metadata.get("url", "")).strip()
        published_date = str(metadata.get("published_date", "")).strip()
        relevance = max(0.0, min(1.0, float(candidate.get("_score", 0.0))))
        authority = _domain_authority_score(url, allowed_suffixes)
        recency = _recency_score(published_date)
        agreement = _agreement_score(candidate, candidates)
        trust = (
            (relevance * relevance_w)
            + (authority * authority_w)
            + (recency * recency_w)
            + (agreement * agreement_w)
        )
        metadata["trust_score"] = round(trust, 4)
        metadata["trust_components"] = {
            "relevance": round(relevance, 4),
            "authority": round(authority, 4),
            "recency": round(recency, 4),
            "agreement": round(agreement, 4),
        }
        candidate["_trust_score"] = trust
        candidate["_final_score"] = (relevance * 0.65) + (trust * 0.35)
    return candidates


async def _asearch_payloads(query_variants: list[str], *, top_k: int) -> list[dict]:
    if not query_variants:
        return []

    search_depth = _search_depth_for_mode()
    num_results = _default_num_for_mode(top_k)
    multi_query_enabled = bool(getattr(settings.web_search, "multi_query_enabled", False))
    if len(query_variants) == 1 and not multi_query_enabled:
        payload = await asearch_google(
            query_variants[0],
            gl=settings.web_search.default_gl,
            hl=settings.web_search.default_hl,
            num=num_results,
            search_depth=search_depth,
        )
        return [payload] if isinstance(payload, dict) else []

    batch = await asearch_google_batch(
        query_variants,
        gl=settings.web_search.default_gl,
        hl=settings.web_search.default_hl,
        num=num_results,
        search_depth=search_depth,
    )

    payloads: list[dict] = []
    first_error = ""
    for item in batch:
        if not isinstance(item, dict):
            continue
        error = str(item.get("error", "")).strip()
        if error and not first_error:
            first_error = error
        payload = item.get("result")
        if isinstance(payload, dict) and payload:
            payloads.append(payload)
    if payloads:
        return payloads
    if first_error:
        raise RuntimeError(first_error)
    return []


def _collect_search_rows(
    payloads: list[dict],
    query_variants: list[str],
    *,
    top_k: int,
    allowed_suffixes: list[str],
) -> list[dict]:
    per_query_limit = _default_num_for_mode(top_k)
    merged_rows: list[dict] = []
    for payload in payloads:
        merged_rows.extend(_organic_rows(payload, limit=per_query_limit))
    dedupe_limit = max(top_k, _max_context_results_for_mode()) * max(1, len(query_variants))
    rows = _dedupe_rows(merged_rows, limit=dedupe_limit)
    return _filter_rows_by_allowed_domains(rows, allowed_suffixes)


def _collect_search_rows_with_domain_retry(
    payloads: list[dict],
    query_variants: list[str],
    *,
    top_k: int,
    allowed_suffixes: list[str],
) -> tuple[list[dict], bool]:
    """Collect rows with strict source filtering and no domain-relax fallback."""
    rows = _collect_search_rows(
        payloads,
        query_variants,
        top_k=top_k,
        allowed_suffixes=allowed_suffixes,
    )
    return rows, False


def _ai_overview_candidate(payloads: list[dict], allowed_suffixes: list[str]) -> dict | None:
    if allowed_suffixes:
        return None
    for payload in payloads:
        ai_text = _ai_overview_text(payload)
        if not ai_text:
            continue
        return {
            "_score": 1.5,
            "chunk_id": "web:ai_overview",
            "source_path": "web_search://google/ai_overview",
            "distance": 0.0,
            "content": ai_text[: settings.web_search.max_page_chars],
            "metadata": {
                "university": "Google AI Overview",
                "title": "Google AI Overview",
                "section_heading": "Web Fallback",
                "url": "",
                "published_date": "",
                "source_type": "google_ai_overview",
            },
        }
    return None


def _page_text_and_date(page_payload) -> tuple[str, str]:
    if isinstance(page_payload, dict):
        page_text = str(page_payload.get("content", "")).strip()
        page_published_date = str(page_payload.get("published_date", "")).strip()
        return page_text, page_published_date
    # Backward-compatible with old test doubles returning a plain page text string.
    return str(page_payload).strip(), ""


def _row_chunk_texts(*, title: str, url: str, snippet: str, page_text: str) -> list[str]:
    chunk_texts = _chunk_clean_text(page_text) if page_text else []
    if snippet and not chunk_texts:
        chunk_texts = [snippet]
    if chunk_texts:
        return chunk_texts
    fallback_text = title or url
    if fallback_text:
        return [fallback_text]
    return []


def _ranked_page_candidates(
    *,
    chunk_texts: list[str],
    query_tokens: set[str],
    snippet: str,
    rank_index: int,
) -> list[tuple[float, int, str]]:
    page_candidates: list[tuple[float, int, str]] = []
    for chunk_index, chunk_text in enumerate(chunk_texts, start=1):
        content = str(chunk_text).strip()
        if not content:
            continue
        score = _chunk_relevance_score(
            query_tokens=query_tokens,
            content=content,
            snippet=snippet,
            rank_index=rank_index,
        )
        page_candidates.append((score, chunk_index, content))
    page_candidates.sort(key=lambda item: item[0], reverse=True)
    return page_candidates


def _organic_row_candidates(
    *,
    index: int,
    row: dict,
    page_data_by_url: dict[str, dict],
    allowed_suffixes: list[str],
    query_tokens: set[str],
) -> list[dict]:
    title = str(row.get("title", "")).strip()
    url = str(row.get("url", "")).strip()
    snippet = str(row.get("snippet", "")).strip()
    if not _source_url_allowed(
        url=url,
        title=title,
        snippet=snippet,
        allowed_suffixes=allowed_suffixes,
    ):
        return []

    row_published_date = str(row.get("published_date", "")).strip()
    page_text, page_published_date = _page_text_and_date(page_data_by_url.get(url, {}))
    published_date = row_published_date or page_published_date

    chunk_texts = _row_chunk_texts(
        title=title,
        url=url,
        snippet=snippet,
        page_text=page_text,
    )
    if not chunk_texts:
        return []

    max_chunks = _max_chunks_per_page_for_mode()
    ranked = _ranked_page_candidates(
        chunk_texts=chunk_texts,
        query_tokens=query_tokens,
        snippet=snippet,
        rank_index=index,
    )
    candidates: list[dict] = []
    for score, chunk_index, content in ranked[:max_chunks]:
        candidates.append(
            {
                "_score": score,
                "chunk_id": f"web:organic:{index}:{chunk_index}",
                "source_path": url or f"web_search://google/organic/{index}",
                "distance": round(max(0.0, 1.0 - min(1.0, score)), 4),
                "content": content[: settings.web_search.max_page_chars],
                "metadata": {
                    "university": title or _host_label(url),
                    "title": title or _host_label(url),
                    "section_heading": "Web Result",
                    "url": url,
                    "published_date": published_date,
                    "source_type": "google_organic",
                },
            }
        )
    return candidates


def _build_organic_candidates(
    *,
    rows: list[dict],
    page_data_by_url: dict[str, dict],
    query_tokens: set[str],
    allowed_suffixes: list[str],
) -> list[dict]:
    candidates: list[dict] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        candidates.extend(
            _organic_row_candidates(
                index=index,
                row=row,
                page_data_by_url=page_data_by_url,
                allowed_suffixes=allowed_suffixes,
                query_tokens=query_tokens,
            )
        )
    return candidates


def _finalize_candidates(candidates: list[dict]) -> list[dict]:
    candidates.sort(
        key=lambda item: float(item.get("_final_score", item.get("_score", 0.0))), reverse=True
    )
    deduped = _dedupe_chunk_candidates(candidates)
    max_results = _max_context_results_for_mode()
    min_unique_domains = _retrieval_min_unique_domains()

    selected_indexes: set[int] = set()
    selected_domains: set[str] = set()
    ordered_items: list[dict] = []

    if min_unique_domains > 1:
        for index, item in enumerate(deduped):
            metadata = item.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            domain = _domain_group_key(_normalized_host(str(metadata.get("url", "")))
            )
            if not domain or domain in selected_domains:
                continue
            selected_domains.add(domain)
            selected_indexes.add(index)
            ordered_items.append(item)
            if len(ordered_items) >= max_results or len(selected_domains) >= min_unique_domains:
                break

    for index, item in enumerate(deduped):
        if index in selected_indexes:
            continue
        ordered_items.append(item)
        if len(ordered_items) >= max_results:
            break

    results: list[dict] = []
    for item in ordered_items[:max_results]:
        cleaned = dict(item)
        cleaned.pop("_score", None)
        cleaned.pop("_trust_score", None)
        cleaned.pop("_final_score", None)
        results.append(cleaned)
    return results


def _fact_text_from_content(content: str) -> str:
    text = " ".join(str(content or "").split()).strip()
    if not text:
        return ""
    sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]
    if not sentences:
        sentences = [text]

    def _score(sentence: str) -> float:
        score = 0.0
        lowered = sentence.lower()
        if _NUMERIC_TOKEN_RE.search(lowered):
            score += 0.9
        if _DATE_VALUE_RE.search(lowered):
            score += 0.7
        if _DEADLINE_CONTENT_RE.search(lowered):
            score += 0.9
        if _LANGUAGE_CONTENT_RE.search(lowered):
            score += 0.8
        if _DURATION_ECTS_CONTENT_RE.search(lowered):
            score += 0.8
        if _ADMISSION_CONTENT_RE.search(lowered):
            score += 0.7
        if _CURRICULUM_CONTENT_RE.search(lowered):
            score += 0.5
        if _TUITION_CONTENT_RE.search(lowered):
            score += 0.6
        if len(sentence) < 25:
            score -= 0.3
        return score

    sentence = max(sentences[:8], key=_score)
    return sentence[:280].strip()


def _extract_facts(candidates: list[dict], *, limit: int) -> list[dict]:
    facts: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        fact_text = _fact_text_from_content(candidate.get("content", ""))
        if not fact_text:
            continue
        key = fact_text.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            {
                "fact": fact_text,
                "url": str(metadata.get("url", "")).strip(),
                "title": str(metadata.get("title", "")).strip(),
                "published_date": str(metadata.get("published_date", "")).strip(),
                "trust_score": float(metadata.get("trust_score", 0.0) or 0.0),
            }
        )
        if len(facts) >= limit:
            break
    return facts


def _normalized_host(url: str) -> str:
    host = str(urlparse(url).hostname or "").strip().lower()
    if host.startswith("www."):
        return host[4:]
    return host


def _unique_domains_from_candidates(candidates: list[dict]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        host = _domain_group_key(_normalized_host(str(metadata.get("url", ""))))
        if not host or host in seen:
            continue
        seen.add(host)
        ordered.append(host)
    return ordered


def _retrieval_min_unique_domains() -> int:
    configured = max(1, int(getattr(settings.web_search, "retrieval_min_unique_domains", 2)))
    if not _is_deep_search_mode():
        return configured
    deep_override = max(1, int(getattr(settings.web_search, "deep_min_unique_domains", configured)))
    return max(configured, deep_override)


def _domain_diversity_gap(unique_domains: list[str]) -> int:
    return max(0, _retrieval_min_unique_domains() - len(unique_domains))


def _domain_gap_subquestions(unique_domains: list[str]) -> list[str]:
    gap = _domain_diversity_gap(unique_domains)
    if gap <= 0:
        return []
    return [f"confirm with at least {gap} additional independent website(s)"]


def _build_domain_gap_queries(query: str, unique_domains: list[str]) -> list[str]:
    gap = _domain_diversity_gap(unique_domains)
    if gap <= 0:
        return []
    candidates: list[str] = []
    seen_domains = {
        _domain_group_key(str(host).strip().lower())
        for host in unique_domains
        if str(host).strip()
    }
    seen_domains.discard("")
    for entity in _comparison_entities_from_query(query):
        entity_focus = _entity_focus_query(entity)
        if not entity_focus:
            continue
        for domain in _official_domains_for_query(entity)[:1]:
            grouped = _domain_group_key(domain)
            if grouped in seen_domains:
                continue
            candidates.append(f"{entity_focus} site:{domain}")
    for domain in _official_domains_for_query(query):
        grouped = _domain_group_key(domain)
        if grouped in seen_domains:
            continue
        candidates.append(f"{query} admission requirements application deadline site:{domain}")
    for _ in range(gap):
        candidates.extend(
            [
                f"{query} official source",
                f"{query} independent source",
                f"{query} corroborated information",
            ]
        )
    return _normalize_query_list(
        candidates,
        limit=max(1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2))),
    )


def _combine_missing_subquestions(base_missing: list[str], unique_domains: list[str]) -> list[str]:
    return _normalize_subquestion_list(
        list(base_missing) + _domain_gap_subquestions(unique_domains),
        limit=max(_max_planner_subquestions(), _retrieval_min_unique_domains() + 2),
    )


def _build_required_field_queries(
    query: str,
    *,
    missing_required_fields: list[dict],
    allowed_suffixes: list[str],
    unique_domains: list[str],
) -> list[str]:
    if not missing_required_fields:
        return []

    candidates: list[str] = []
    seen_domains = {
        _domain_group_key(str(host).strip().lower())
        for host in unique_domains
        if str(host).strip()
    }
    seen_domains.discard("")

    official_domains = _official_domains_for_query(query)[:3]
    suffix_scope = ""
    if allowed_suffixes:
        suffix_scope = " (" + " OR ".join(f"site:{suffix}" for suffix in allowed_suffixes[:2]) + ")"

    for field in missing_required_fields:
        focus = " ".join(str(field.get("query_focus", "")).split()).strip()
        if not focus:
            continue
        candidates.append(f"{query} {focus} official source")
        if suffix_scope:
            candidates.append(f"{query} {focus}{suffix_scope}")
        for domain in official_domains:
            grouped = _domain_group_key(domain)
            if grouped in seen_domains:
                continue
            candidates.append(f"{query} {focus} site:{domain}")

    max_gap_queries = max(1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2)))
    return _normalize_query_list(
        candidates,
        limit=max_gap_queries * 3,
    )


def _build_follow_up_queries(
    query: str,
    *,
    missing_subquestions: list[str],
    llm_gap_queries: list[str],
    missing_required_fields: list[dict],
    allowed_suffixes: list[str],
    unique_domains: list[str],
) -> list[str]:
    max_gap_queries = max(1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2)))
    candidate_limit = max_gap_queries * 4
    required_field_queries = _build_required_field_queries(
        query,
        missing_required_fields=missing_required_fields,
        allowed_suffixes=allowed_suffixes,
        unique_domains=unique_domains,
    )
    heuristic_queries = _build_gap_queries(query, missing_subquestions)
    domain_queries = _build_domain_gap_queries(query, unique_domains)
    if llm_gap_queries:
        return _normalize_query_list(
            required_field_queries + llm_gap_queries + domain_queries,
            limit=candidate_limit,
        )
    return _normalize_query_list(
        required_field_queries + heuristic_queries + domain_queries,
        limit=candidate_limit,
    )


def _next_loop_queries(
    *,
    base_query: str,
    initial_queries: list[str],
    first_wave_queries: list[str],
    missing_subquestions: list[str],
    llm_gap_queries: list[str],
    follow_up_queries: list[str],
    seen_queries: set[str],
    loop_step: int,
    deep_mode: bool,
) -> list[str]:
    if loop_step <= 1:
        if not deep_mode:
            return _next_queries_for_loop(
                initial_queries,
                seen_queries,
                max_queries=min(2, _max_planner_queries()),
            )
        first_wave_limit = max(
            _max_planner_queries(),
            _max_planner_queries() + max(
                1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2))
            ),
        )
        first_wave_queries = _normalize_query_list(
            list(initial_queries) + list(first_wave_queries),
            limit=first_wave_limit,
        )
        return _next_queries_for_loop(
            first_wave_queries,
            seen_queries,
            max_queries=first_wave_limit,
        )
    gap_queries = (
        llm_gap_queries or follow_up_queries or _build_gap_queries(base_query, missing_subquestions)
    )
    return _next_queries_for_loop(
        gap_queries,
        seen_queries,
        max_queries=max(1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2))),
    )


def _retrieval_loop_enabled() -> bool:
    return _is_deep_search_mode() and bool(getattr(settings.web_search, "retrieval_loop_enabled", True))


async def aretrieve_web_chunks(
    query: str,
    *,
    top_k: int = 3,
    search_mode: str = "deep",
) -> dict:
    normalized_mode = _normalized_search_mode(search_mode)
    mode_token = _RETRIEVAL_MODE_CTX.set(normalized_mode)
    try:
        return await _aretrieve_web_chunks_impl(
            query,
            top_k=top_k,
            search_mode=normalized_mode,
        )
    finally:
        _RETRIEVAL_MODE_CTX.reset(mode_token)


async def _aretrieve_web_chunks_impl(
    query: str,
    *,
    top_k: int = 3,
    search_mode: str = "deep",
) -> dict:
    """Retrieve fallback web evidence from Tavily."""
    started_at = time.perf_counter()
    allowed_suffixes = _normalized_allowed_domain_suffixes()
    normalized_mode = _normalized_search_mode(search_mode)
    deep_mode = _is_deep_search_mode(normalized_mode)
    required_fields = _required_fields_from_query(query) if deep_mode else []
    if deep_mode:
        query_plan = await _resolve_query_plan(query, allowed_suffixes)
    else:
        query_plan = _build_heuristic_query_plan(query, allowed_suffixes)
        query_plan["planner"] = "fast_heuristic"
        query_plan["llm_used"] = False
        query_plan["subquestions"] = []
    emit_trace_event(
        "query_plan_created",
        {
            "query": query[:220],
            "search_mode": normalized_mode,
            "planner": str(query_plan.get("planner", "heuristic")),
            "llm_used": bool(query_plan.get("llm_used", False)),
            "subquestions": query_plan.get("subquestions", []),
            "required_fields": [str(field.get("id", "")).strip() for field in required_fields],
            "queries": query_plan.get("queries", []),
        },
    )
    planned_query_limit = _max_planner_queries() if deep_mode else min(2, _max_planner_queries())
    planned_queries = _normalize_query_list(
        query_plan.get("queries", []),
        limit=planned_query_limit,
    ) or _build_query_variants(query, allowed_suffixes)
    if deep_mode:
        subquestions = _normalize_subquestion_list(
            query_plan.get("subquestions", []),
            limit=_max_planner_subquestions(),
        )
    else:
        subquestions = []

    search_ms_total = 0
    fetch_ms_total = 0
    domain_filter_relaxed = False
    all_candidates: list[dict] = []
    all_facts: list[dict] = []
    gap_iterations: list[dict] = []
    seen_queries: set[str] = set()
    executed_queries: list[str] = []
    loop_llm_used = False
    max_steps = max(1, int(getattr(settings.web_search, "retrieval_loop_max_steps", 2)))
    if not deep_mode or not _retrieval_loop_enabled():
        max_steps = 1

    for step in range(1, max_steps + 1):
        current_domains = _unique_domains_from_candidates(all_candidates)
        heuristic_missing = _identify_missing_subquestions(subquestions, all_facts)
        required_status = _required_field_coverage(required_fields, all_candidates)
        missing_required_fields = _required_fields_by_ids(
            required_fields,
            required_status.get("missing_ids", []),
        )
        missing_subquestions = _combine_missing_subquestions(
            list(heuristic_missing) + list(required_status.get("missing_subquestions", [])),
            current_domains,
        )
        llm_gap_queries: list[str] = []
        follow_up_queries = _build_follow_up_queries(
            query,
            missing_subquestions=missing_subquestions,
            llm_gap_queries=llm_gap_queries,
            missing_required_fields=missing_required_fields,
            allowed_suffixes=allowed_suffixes,
            unique_domains=current_domains,
        )
        if deep_mode and step > 1 and missing_subquestions:
            llm_gap_plan = await _aidentify_gap_plan_with_llm(
                query,
                subquestions=subquestions or missing_subquestions,
                facts=all_facts,
                fallback_missing=missing_subquestions,
            )
            if llm_gap_plan:
                loop_llm_used = True
                missing_subquestions = _normalize_subquestion_list(
                    list(llm_gap_plan.get("missing_subquestions", []))
                    + list(required_status.get("missing_subquestions", [])),
                    limit=max(_max_planner_subquestions(), _retrieval_min_unique_domains() + 2),
                ) or list(missing_subquestions)
                llm_gap_queries = _normalize_query_list(
                    llm_gap_plan.get("queries", []),
                    limit=max(
                        1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2))
                    ),
                )
                follow_up_queries = _build_follow_up_queries(
                    query,
                    missing_subquestions=missing_subquestions,
                    llm_gap_queries=llm_gap_queries,
                    missing_required_fields=missing_required_fields,
                    allowed_suffixes=allowed_suffixes,
                    unique_domains=current_domains,
                )

        loop_queries = _next_loop_queries(
            base_query=query,
            initial_queries=planned_queries,
            first_wave_queries=(
                _normalize_query_list(
                    _build_gap_queries(query, subquestions)
                    + _build_required_field_queries(
                        query,
                        missing_required_fields=required_fields,
                        allowed_suffixes=allowed_suffixes,
                        unique_domains=current_domains,
                    ),
                    limit=max(
                        _max_planner_queries(),
                        _max_planner_queries()
                        + max(1, int(getattr(settings.web_search, "retrieval_loop_max_gap_queries", 2))),
                    ),
                )
                if deep_mode and bool(getattr(settings.web_search, "multi_query_enabled", False))
                else []
            ),
            missing_subquestions=missing_subquestions,
            llm_gap_queries=llm_gap_queries,
            follow_up_queries=follow_up_queries,
            seen_queries=seen_queries,
            loop_step=step,
            deep_mode=deep_mode,
        )
        if not loop_queries:
            break
        executed_queries.extend(loop_queries)
        emit_trace_event(
            "search_started",
            {
                "step": step,
                "queries": loop_queries,
            },
        )

        search_started_at = time.perf_counter()
        payloads = await _asearch_payloads(loop_queries, top_k=top_k)
        search_ms_total += _elapsed_ms(search_started_at)

        rows, relaxed = _collect_search_rows_with_domain_retry(
            payloads,
            loop_queries,
            top_k=top_k,
            allowed_suffixes=allowed_suffixes,
        )
        domain_filter_relaxed = domain_filter_relaxed or relaxed
        emit_trace_event(
            "search_results",
            {
                "step": step,
                "result_count": len(rows),
                "urls": [str(row.get("url", "")).strip() for row in rows[:8]],
                "domain_filter_relaxed": relaxed,
            },
        )

        fetch_started_at = time.perf_counter()
        if deep_mode:
            page_data_by_url = await _afetch_organic_pages(rows)
        else:
            page_data_by_url = await _afetch_organic_pages(rows, max_pages_to_fetch=1)
        fetch_ms_total += _elapsed_ms(fetch_started_at)
        emit_trace_event(
            "pages_read",
            {
                "step": step,
                "pages_fetched": len(page_data_by_url),
                "urls": list(page_data_by_url.keys())[:8],
            },
        )

        query_tokens = _query_tokens(" ".join(loop_queries))
        candidates = _build_organic_candidates(
            rows=rows,
            page_data_by_url=page_data_by_url,
            query_tokens=query_tokens,
            allowed_suffixes=allowed_suffixes,
        )
        ai_candidate = _ai_overview_candidate(payloads, allowed_suffixes)
        if ai_candidate:
            candidates.append(ai_candidate)

        candidates = _apply_trust_scores(candidates, allowed_suffixes)
        all_candidates.extend(candidates)
        all_candidates = _apply_trust_scores(all_candidates, allowed_suffixes)
        all_facts = _extract_facts(
            all_candidates,
            limit=(
                max(2, _max_context_results_for_mode() * 3)
                if deep_mode
                else max(2, _max_context_results_for_mode())
            ),
        )
        emit_trace_event(
            "facts_extracted",
            {
                "step": step,
                "fact_count": len(all_facts),
                "facts": all_facts[:5],
            },
        )

        unique_domains = _unique_domains_from_candidates(all_candidates)
        next_heuristic_missing = _identify_missing_subquestions(subquestions, all_facts)
        next_required_status = _required_field_coverage(required_fields, all_candidates)
        next_missing_required_ids = list(next_required_status.get("missing_ids", []))
        next_missing_required_fields = _required_fields_by_ids(required_fields, next_missing_required_ids)
        next_missing = _combine_missing_subquestions(
            list(next_heuristic_missing) + list(next_required_status.get("missing_subquestions", [])),
            unique_domains,
        )
        next_follow_up_queries = _build_follow_up_queries(
            query,
            missing_subquestions=next_missing,
            llm_gap_queries=llm_gap_queries,
            missing_required_fields=next_missing_required_fields,
            allowed_suffixes=allowed_suffixes,
            unique_domains=unique_domains,
        )
        retrieval_verified = (
            (
                len(unique_domains) >= _retrieval_min_unique_domains()
                and not next_missing
                and not next_missing_required_ids
            )
            if deep_mode
            else True
        )
        emit_trace_event(
            "retrieval_verification",
            {
                "step": step,
                "verified": retrieval_verified,
                "search_mode": normalized_mode,
                "min_unique_domains": _retrieval_min_unique_domains(),
                "unique_domain_count": len(unique_domains),
                "unique_domains": unique_domains[:8],
                "missing_subquestions": next_missing,
                "required_field_coverage": round(
                    float(next_required_status.get("coverage", 1.0) or 0.0),
                    4,
                ),
                "required_fields_missing": next_missing_required_ids,
            },
        )
        gap_iterations.append(
            {
                "step": step,
                "queries": loop_queries,
                "llm_gap_queries": llm_gap_queries,
                "follow_up_queries": next_follow_up_queries,
                "missing_subquestions": next_missing,
                "required_field_coverage": round(
                    float(next_required_status.get("coverage", 1.0) or 0.0),
                    4,
                ),
                "required_fields_missing": next_missing_required_ids,
                "unique_domains": unique_domains,
                "unique_domain_count": len(unique_domains),
            }
        )
        emit_trace_event(
            "gaps_identified",
            {
                "step": step,
                "missing_subquestions": next_missing,
                "follow_up_queries": next_follow_up_queries,
            },
        )
        if retrieval_verified:
            break

    results = _finalize_candidates(all_candidates)
    extracted_facts = _extract_facts(
        results,
        limit=_max_context_results_for_mode(),
    )
    final_domains = _unique_domains_from_candidates(results)
    final_required_field_status = _required_field_coverage(required_fields, results)
    final_missing_subquestions = _combine_missing_subquestions(
        list(_identify_missing_subquestions(subquestions, extracted_facts))
        + list(final_required_field_status.get("missing_subquestions", [])),
        final_domains,
    )
    final_missing_required_ids = list(final_required_field_status.get("missing_ids", []))
    final_verified = (
        (
            len(final_domains) >= _retrieval_min_unique_domains()
            and not final_missing_subquestions
            and not final_missing_required_ids
        )
        if deep_mode
        else bool(results)
    )
    emit_trace_event(
        "source_ranking_completed",
        {
            "result_count": len(results),
            "facts": extracted_facts,
            "unique_domain_count": len(final_domains),
            "unique_domains": final_domains[:8],
            "required_field_coverage": round(
                float(final_required_field_status.get("coverage", 1.0) or 0.0),
                4,
            ),
            "required_fields_missing": final_missing_required_ids,
            "urls": [
                str((item.get("metadata") or {}).get("url", "")).strip()
                for item in results[:8]
                if isinstance(item, dict)
            ],
        },
    )

    return {
        "query": query,
        "query_variants": executed_queries,
        "search_mode": normalized_mode,
        "query_plan": {
            "planner": str(query_plan.get("planner", "heuristic")),
            "llm_used": bool(query_plan.get("llm_used", False)),
            "subquestions": subquestions,
            "required_fields": [str(field.get("id", "")).strip() for field in required_fields],
        },
        "retrieval_loop": {
            "enabled": bool(deep_mode and _retrieval_loop_enabled()),
            "llm_used": loop_llm_used,
            "iterations": len(gap_iterations),
            "steps": gap_iterations,
        },
        "verification": {
            "min_unique_domains": _retrieval_min_unique_domains(),
            "unique_domains": final_domains,
            "unique_domain_count": len(final_domains),
            "missing_subquestions": final_missing_subquestions,
            "required_field_coverage": round(
                float(final_required_field_status.get("coverage", 1.0) or 0.0),
                4,
            ),
            "required_fields": final_required_field_status.get("fields", []),
            "required_fields_missing": final_missing_required_ids,
            "required_field_labels_missing": final_required_field_status.get("missing_labels", []),
            "verified": final_verified,
        },
        "facts": extracted_facts,
        "retrieval_strategy": (
            "web_search_domain_relaxed" if domain_filter_relaxed else "web_search"
        ),
        "domain_filter_relaxed": domain_filter_relaxed,
        "timings_ms": {
            "search": search_ms_total,
            "page_fetch": fetch_ms_total,
            "total": _elapsed_ms(started_at),
        },
        "results": results,
    }
