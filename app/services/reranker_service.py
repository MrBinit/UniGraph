import json
import logging
import time

import boto3

from app.core.config import get_settings
from app.infra.bedrock_client import ainvoke_model_json
from app.infra.io_limiters import dependency_limiter

settings = get_settings()
logger = logging.getLogger(__name__)

_JSON_CONTENT_TYPE = "application/json"


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _normalized_query(query: str) -> str:
    return " ".join(str(query).split())[: settings.bedrock.reranker_max_query_chars]


def _content_and_metadata(result: dict) -> tuple[str, dict]:
    if not isinstance(result, dict):
        return "", {}
    content = str(result.get("content", "")).strip()
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return content, metadata


def _document_text(result: dict) -> str:
    content, metadata = _content_and_metadata(result)
    if not content:
        return ""

    heading = str(metadata.get("section_heading", "")).strip()
    title = str(metadata.get("university", "")).strip()
    url = str(metadata.get("url", "")).strip()
    prefix = " | ".join(part for part in (title, heading, url) if part).strip()
    merged = f"{prefix} | {content}".strip(" |") if prefix else content
    return merged[: settings.bedrock.reranker_max_document_chars]


def _normalize_candidates(candidates: list[dict]) -> tuple[list[dict], list[str]]:
    usable_results: list[dict] = []
    documents: list[str] = []
    max_docs = max(1, int(settings.bedrock.reranker_max_documents))
    for result in candidates:
        if len(usable_results) >= max_docs:
            break
        if not isinstance(result, dict):
            continue
        document = _document_text(result)
        if not document:
            continue
        usable_results.append(result)
        documents.append(document)
    return usable_results, documents


def _parse_ranked_indices(payload: dict, total_docs: int) -> list[tuple[int, float]]:
    ranked: list[tuple[int, float]] = []
    seen: set[int] = set()
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return ranked
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("index", -1))
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= total_docs or index in seen:
            continue
        seen.add(index)
        try:
            score = float(row.get("relevance_score", row.get("score", 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        ranked.append((index, score))
    return ranked


def _needs_cohere_api_version(model_id: str) -> bool:
    return str(model_id).strip().lower().startswith("cohere.rerank")


def _reranker_request_body(
    *, query_text: str, documents: list[str], top_n: int, model_id: str
) -> dict:
    body = {
        "query": query_text,
        "documents": documents,
        "top_n": top_n,
    }
    if _needs_cohere_api_version(model_id):
        body["api_version"] = int(getattr(settings.bedrock, "reranker_api_version", 2))
    return body


def _reranker_enabled() -> bool:
    if not settings.bedrock.reranker_enabled:
        return False
    if not str(settings.bedrock.reranker_model_id).strip():
        return False
    try:
        return boto3.session.Session().get_credentials() is not None
    except Exception:
        return False


async def arerank_retrieval_results(query: str, candidates: list[dict]) -> dict:
    """Rerank retrieval candidates with Bedrock Cohere Rerank and return top results."""
    started_at = time.perf_counter()
    if not _reranker_enabled():
        return {
            "results": candidates,
            "applied": False,
            "timings_ms": {"total": _elapsed_ms(started_at)},
        }

    query_text = _normalized_query(query)
    if not query_text:
        return {
            "results": candidates,
            "applied": False,
            "timings_ms": {"total": _elapsed_ms(started_at)},
        }

    normalized_results, documents = _normalize_candidates(candidates or [])
    min_docs = max(1, int(settings.bedrock.reranker_min_documents))
    if len(normalized_results) < min_docs:
        return {
            "results": normalized_results,
            "applied": False,
            "timings_ms": {"total": _elapsed_ms(started_at)},
        }

    top_n = min(
        max(1, int(settings.bedrock.reranker_top_n)),
        len(normalized_results),
    )
    model_id = str(settings.bedrock.reranker_model_id).strip()
    request_payload = {
        "modelId": model_id,
        "body": json.dumps(
            _reranker_request_body(
                query_text=query_text,
                documents=documents,
                top_n=top_n,
                model_id=model_id,
            )
        ),
        "contentType": _JSON_CONTENT_TYPE,
        "accept": _JSON_CONTENT_TYPE,
    }

    invoke_started_at = time.perf_counter()
    async with dependency_limiter("reranker"):
        response_payload = await ainvoke_model_json(request_payload)
    invoke_ms = _elapsed_ms(invoke_started_at)

    ranked_indices = _parse_ranked_indices(response_payload, total_docs=len(normalized_results))
    if not ranked_indices:
        logger.warning("Reranker returned no ranked results; using original order.")
        return {
            "results": normalized_results[:top_n],
            "applied": False,
            "timings_ms": {
                "model": invoke_ms,
                "total": _elapsed_ms(started_at),
            },
        }

    reranked: list[dict] = []
    for index, score in ranked_indices:
        item = dict(normalized_results[index])
        item["rerank_score"] = score
        reranked.append(item)
        if len(reranked) >= top_n:
            break

    return {
        "results": reranked,
        "applied": True,
        "timings_ms": {
            "model": invoke_ms,
            "total": _elapsed_ms(started_at),
        },
    }
