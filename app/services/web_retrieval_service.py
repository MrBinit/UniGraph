import asyncio
import re
import time
import urllib.request
from html import unescape
from urllib.parse import urlparse

from app.core.config import get_settings
from app.infra.io_limiters import dependency_limiter
from app.services.serpapi_search_service import asearch_google, asearch_google_batch

settings = get_settings()

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


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


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
    if settings.serpapi.strip_boilerplate:
        text = _BOILERPLATE_BLOCK_RE.sub("\n", text)
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = unescape(text).replace("\xa0", " ")
    min_line_chars = max(0, int(settings.serpapi.min_clean_line_chars))

    lines: list[str] = []
    used_chars = 0
    for raw_line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line).strip(" |-\t")
        if not line:
            continue
        if min_line_chars and len(line) < min_line_chars:
            continue
        if settings.serpapi.strip_boilerplate and _is_boilerplate_line(line):
            continue
        lines.append(line)
        used_chars += len(line) + 1
        if used_chars >= max_chars:
            break
    return "\n".join(lines)[:max_chars]


def _fetch_page_data_sync(url: str, timeout_seconds: float, max_chars: int) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "unigraph-web-retrieval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        content_type = str(response.headers.get("Content-Type", "")).lower()
        if "text/html" not in content_type:
            return {"content": "", "published_date": ""}
        raw = response.read(max_chars * 8).decode("utf-8", errors="ignore")

    return {
        "content": _clean_html_text(raw, max_chars=max_chars),
        "published_date": _extract_published_date(raw),
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
    async with dependency_limiter("serpapi"):
        return await asyncio.to_thread(
            _fetch_page_data_sync,
            url,
            float(settings.serpapi.page_fetch_timeout_seconds),
            int(settings.serpapi.max_page_chars),
        )


async def _afetch_organic_pages(rows: list[dict]) -> dict[str, dict]:
    if not settings.serpapi.fetch_page_content:
        return {}

    targets = [row for row in rows if row.get("url")]
    targets = targets[: settings.serpapi.max_pages_to_fetch]
    if not targets:
        return {}

    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.serpapi.queue_max_size)
    worker_count = min(settings.serpapi.queue_workers, len(targets))
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
    raw = getattr(settings.serpapi, "allowed_domain_suffixes", [])
    if not isinstance(raw, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        suffix = str(entry).strip().lower()
        if not suffix:
            continue
        if not suffix.startswith("."):
            suffix = f".{suffix.lstrip('.')}"
        if suffix in seen:
            continue
        seen.add(suffix)
        normalized.append(suffix)
    return normalized


def _build_query_variants(query: str, allowed_suffixes: list[str]) -> list[str]:
    base = " ".join(str(query).split()).strip()
    if not base:
        return []

    if not settings.serpapi.multi_query_enabled:
        return [base]

    candidates: list[str] = [base, f"{base} official information"]
    compact = _compact_query_keywords(base)
    if compact and compact != base.lower():
        candidates.append(compact)

    if allowed_suffixes:
        site_terms = " OR ".join(f"site:{suffix}" for suffix in allowed_suffixes[:2])
        candidates.append(f"{base} ({site_terms})")

    max_variants = max(1, int(settings.serpapi.max_query_variants))
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


def _url_matches_allowed_suffix(url: str, allowed_suffixes: list[str]) -> bool:
    if not allowed_suffixes:
        return True
    host = str(urlparse(url).hostname or "").strip().lower()
    if not host:
        return False
    return any(host.endswith(suffix) for suffix in allowed_suffixes)


def _filter_rows_by_allowed_domains(rows: list[dict], allowed_suffixes: list[str]) -> list[dict]:
    if not allowed_suffixes:
        return rows
    return [
        row
        for row in rows
        if _url_matches_allowed_suffix(str(row.get("url", "")), allowed_suffixes)
    ]


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
    max_chars = max(120, int(settings.serpapi.page_chunk_chars))
    overlap = max(0, min(int(settings.serpapi.page_chunk_overlap_chars), max_chars // 2))
    max_chunks = max(1, int(settings.serpapi.max_chunks_per_page))
    min_chunk_chars = max(20, int(settings.serpapi.min_chunk_chars))
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
    threshold = float(settings.serpapi.chunk_dedupe_similarity)
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


async def _asearch_payloads(query_variants: list[str], *, top_k: int) -> list[dict]:
    if not query_variants:
        return []

    if len(query_variants) == 1:
        payload = await asearch_google(
            query_variants[0],
            gl=settings.serpapi.default_gl,
            hl=settings.serpapi.default_hl,
            num=max(top_k, settings.serpapi.default_num),
        )
        return [payload] if isinstance(payload, dict) else []

    batch = await asearch_google_batch(
        query_variants,
        gl=settings.serpapi.default_gl,
        hl=settings.serpapi.default_hl,
        num=max(top_k, settings.serpapi.default_num),
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
    per_query_limit = max(top_k, settings.serpapi.default_num)
    merged_rows: list[dict] = []
    for payload in payloads:
        merged_rows.extend(_organic_rows(payload, limit=per_query_limit))
    dedupe_limit = max(top_k, settings.serpapi.max_context_results) * max(1, len(query_variants))
    rows = _dedupe_rows(merged_rows, limit=dedupe_limit)
    return _filter_rows_by_allowed_domains(rows, allowed_suffixes)


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
            "source_path": "serpapi://google/ai_overview",
            "distance": 0.0,
            "content": ai_text[: settings.serpapi.max_page_chars],
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
    if not _url_matches_allowed_suffix(url, allowed_suffixes):
        return []

    snippet = str(row.get("snippet", "")).strip()
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

    max_chunks = max(1, int(settings.serpapi.max_chunks_per_page))
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
                "source_path": url or f"serpapi://google/organic/{index}",
                "distance": round(max(0.0, 1.0 - min(1.0, score)), 4),
                "content": content[: settings.serpapi.max_page_chars],
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
    candidates.sort(key=lambda item: float(item.get("_score", 0.0)), reverse=True)
    deduped = _dedupe_chunk_candidates(candidates)

    results: list[dict] = []
    for item in deduped:
        cleaned = dict(item)
        cleaned.pop("_score", None)
        results.append(cleaned)
        if len(results) >= settings.serpapi.max_context_results:
            break
    return results


async def aretrieve_web_chunks(query: str, *, top_k: int = 3) -> dict:
    """Retrieve fallback web evidence from Google via SerpAPI."""
    started_at = time.perf_counter()
    search_started_at = time.perf_counter()
    allowed_suffixes = _normalized_allowed_domain_suffixes()
    query_variants = _build_query_variants(query, allowed_suffixes)
    payloads = await _asearch_payloads(query_variants, top_k=top_k)
    search_ms = _elapsed_ms(search_started_at)

    rows = _collect_search_rows(
        payloads,
        query_variants,
        top_k=top_k,
        allowed_suffixes=allowed_suffixes,
    )
    fetch_started_at = time.perf_counter()
    page_data_by_url = await _afetch_organic_pages(rows)
    fetch_ms = _elapsed_ms(fetch_started_at)

    query_tokens = _query_tokens(query)
    candidates = _build_organic_candidates(
        rows=rows,
        page_data_by_url=page_data_by_url,
        query_tokens=query_tokens,
        allowed_suffixes=allowed_suffixes,
    )
    ai_candidate = _ai_overview_candidate(payloads, allowed_suffixes)
    if ai_candidate:
        candidates.append(ai_candidate)
    results = _finalize_candidates(candidates)

    return {
        "query": query,
        "query_variants": query_variants,
        "retrieval_strategy": "web_search",
        "timings_ms": {
            "search": search_ms,
            "page_fetch": fetch_ms,
            "total": _elapsed_ms(started_at),
        },
        "results": results,
    }
