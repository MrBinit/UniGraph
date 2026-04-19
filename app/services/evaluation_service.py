from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from statistics import mean
from uuid import uuid4

try:
    import boto3
    from boto3.dynamodb.types import TypeDeserializer
except Exception:  # pragma: no cover - boto3 optional in minimal test envs
    boto3 = None
    TypeDeserializer = None
try:
    from botocore.config import Config as BotoConfig
except Exception:  # pragma: no cover - botocore optional in minimal test envs
    BotoConfig = None

from app.core.config import get_settings
from app.core.memory_crypto import decrypt_memory_payload, encrypt_memory_payload
from app.infra.redis_client import app_scoped_key, redis_client
from app.services.quality_metrics_service import (
    aggregate_metric_rows,
    citation_accuracy_score,
    generation_metrics,
    retrieval_metrics,
)

settings = get_settings()
logger = logging.getLogger(__name__)
_ddb_deserializer = TypeDeserializer() if TypeDeserializer is not None else None

MAX_STORED_CONVERSATIONS = 500


def _conversation_key(conversation_id: str) -> str:
    return app_scoped_key("eval", "conversation", conversation_id)


def _user_conversation_index_key(user_id: str) -> str:
    return app_scoped_key("eval", "user", user_id, "conversations")


def _serialize_trace(trace: dict) -> str:
    return encrypt_memory_payload(trace)


def _deserialize_trace(raw: str) -> dict | None:
    parsed = decrypt_memory_payload(raw)
    return parsed if isinstance(parsed, dict) else None


def _safe_payload_results(results: list[dict] | None) -> list[dict]:
    safe_results: list[dict] = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        safe_results.append(
            {
                "chunk_id": result.get("chunk_id", ""),
                "document_id": result.get("document_id", ""),
                "source_path": result.get("source_path", ""),
                "metadata": result.get("metadata", {}),
                "content": result.get("content", ""),
                "distance": result.get("distance"),
            }
        )
    return safe_results


def _dynamodb_region_name() -> str | None:
    return (
        os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
        or None
    )


def _dynamodb_client():
    if boto3 is None:
        return None
    kwargs = {"region_name": _dynamodb_region_name()} if _dynamodb_region_name() else {}
    if BotoConfig is not None:
        kwargs["config"] = BotoConfig(
            connect_timeout=1,
            read_timeout=3,
            retries={"max_attempts": 1, "mode": "standard"},
        )
    return boto3.client("dynamodb", **kwargs)


def _deserialize_dynamodb_item(item: dict) -> dict:
    if not isinstance(item, dict) or _ddb_deserializer is None:
        return {}
    try:
        return {key: _ddb_deserializer.deserialize(value) for key, value in item.items()}
    except Exception:
        return {}


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _evidence_urls_from_metrics_row(row: dict) -> list[str]:
    raw = str(row.get("retrieval_evidence_json", "")).strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("source_path", "")).strip()
        if not candidate:
            metadata = item.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            candidate = str(metadata.get("url", "")).strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        urls.append(candidate)
        if len(urls) >= 12:
            break
    return urls


def _trace_from_metrics_row(row: dict) -> dict | None:
    if not isinstance(row, dict):
        return None
    prompt = str(row.get("query", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not prompt and not answer:
        return None
    created_at = str(row.get("timestamp", "")).strip() or datetime.now(timezone.utc).isoformat()
    conversation_id = str(row.get("request_id", "")).strip() or uuid4().hex
    retrieval_result_count = max(0, _coerce_int(row.get("retrieval_result_count"), 0))
    evidence_urls = _evidence_urls_from_metrics_row(row)
    reconstructed_results = [
        {
            "chunk_id": f"ddb:{index}",
            "source_path": url,
            "metadata": {"url": url},
            "content": "",
        }
        for index, url in enumerate(evidence_urls, start=1)
    ]
    return {
        "conversation_id": conversation_id,
        "user_id": str(row.get("user_id", "")).strip(),
        "prompt": prompt,
        "answer": answer,
        "created_at": created_at,
        "retrieval_strategy": str(row.get("retrieval_strategy", "")).strip(),
        "retrieved_results": reconstructed_results,
        "retrieval_result_count": retrieval_result_count,
        "quality": {
            "groundedness": _coerce_float(row.get("groundedness"), 0.0),
            "citation_accuracy": _coerce_float(row.get("citation_accuracy"), 0.0),
        },
        "evidence_urls": evidence_urls,
        "labels": {
            "expected_answer": None,
            "relevant_chunk_ids": [],
            "user_feedback": None,
            "user_feedback_score": None,
        },
        "_trace_source": "dynamodb_metrics",
    }


def _list_chat_traces_from_dynamodb(user_id: str, *, limit: int = 20) -> list[dict]:
    if limit <= 0:
        return []
    if not bool(settings.app.metrics_dynamodb_enabled):
        return []
    table_name = str(settings.app.metrics_dynamodb_requests_table or "").strip()
    if not table_name:
        return []
    client = _dynamodb_client()
    if client is None:
        return []

    max_pages = 2
    page_size = min(200, max(25, limit * 3))
    deadline = time.monotonic() + 5.0
    collected_items: list[dict] = []
    start_key = None
    for _ in range(max_pages):
        if time.monotonic() >= deadline:
            break
        params = {
            "TableName": table_name,
            "Limit": page_size,
            "FilterExpression": "#uid = :uid",
            "ExpressionAttributeNames": {
                "#uid": "user_id",
                "#ts": "timestamp",
                "#query": "query",
                "#ans": "answer",
            },
            "ExpressionAttributeValues": {
                ":uid": {"S": str(user_id)},
            },
            "ProjectionExpression": (
                "request_id,#ts,#uid,#query,#ans,retrieval_strategy,retrieval_result_count,"
                "groundedness,citation_accuracy,retrieval_evidence_json,outcome"
            ),
        }
        if start_key:
            params["ExclusiveStartKey"] = start_key
        try:
            response = client.scan(**params)
        except Exception as exc:
            logger.warning("DynamoDB scan for chat traces failed for user=%s. %s", user_id, exc)
            break
        items = response.get("Items", [])
        if isinstance(items, list):
            collected_items.extend(items)
        start_key = response.get("LastEvaluatedKey")
        if not start_key:
            break
        if len(collected_items) >= max(limit * 5, 120):
            break

    rows = [_deserialize_dynamodb_item(item) for item in collected_items]
    filtered_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("user_id", "")).strip() != str(user_id):
            continue
        outcome = str(row.get("outcome", "")).strip().lower()
        if outcome and outcome != "success":
            continue
        if not str(row.get("answer", "")).strip():
            continue
        filtered_rows.append(row)
    filtered_rows.sort(key=lambda row: str(row.get("timestamp", "")).strip(), reverse=True)

    traces: list[dict] = []
    seen_ids: set[str] = set()
    for row in filtered_rows:
        trace = _trace_from_metrics_row(row)
        if not trace:
            continue
        conversation_id = str(trace.get("conversation_id", "")).strip()
        if not conversation_id or conversation_id in seen_ids:
            continue
        seen_ids.add(conversation_id)
        traces.append(trace)
        if len(traces) >= limit:
            break
    return traces


def store_chat_trace(
    *,
    user_id: str,
    prompt: str,
    answer: str,
    retrieved_results: list[dict] | None,
    retrieval_strategy: str,
    timings_ms: dict | None,
    quality: dict | None = None,
    evidence_urls: list[str] | None = None,
    redis=None,
) -> str | None:
    """
    Persist a chat trace so evaluation can run on real question/answer traffic.

    Failures are non-fatal to the chat pipeline.
    """
    client = redis or redis_client
    conversation_id = uuid4().hex
    now_iso = datetime.now(timezone.utc).isoformat()
    trace = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "prompt": prompt,
        "answer": answer,
        "created_at": now_iso,
        "retrieval_strategy": retrieval_strategy,
        "timings_ms": timings_ms or {},
        "retrieved_results": _safe_payload_results(retrieved_results),
        "quality": quality if isinstance(quality, dict) else {},
        "evidence_urls": [
            str(url).strip()
            for url in (evidence_urls or [])
            if isinstance(url, str) and str(url).strip()
        ],
        "labels": {
            "expected_answer": None,
            "relevant_chunk_ids": [],
            "user_feedback": None,
            "user_feedback_score": None,
        },
    }

    ttl_seconds = settings.memory.redis_ttl_seconds
    trace_key = _conversation_key(conversation_id)
    user_index_key = _user_conversation_index_key(user_id)

    try:
        client.setex(trace_key, ttl_seconds, _serialize_trace(trace))
        client.lpush(user_index_key, conversation_id)
        client.ltrim(user_index_key, 0, MAX_STORED_CONVERSATIONS - 1)
        client.expire(user_index_key, ttl_seconds)
        return conversation_id
    except Exception as exc:
        logger.warning("Failed to persist chat evaluation trace for user=%s. %s", user_id, exc)
        return None


def _load_trace(conversation_id: str, *, redis=None) -> dict | None:
    client = redis or redis_client
    trace_key = _conversation_key(conversation_id)
    try:
        raw = client.get(trace_key)
    except Exception:
        return None
    if not raw:
        return None
    return _deserialize_trace(raw)


def list_chat_traces(user_id: str, *, limit: int = 20, redis=None) -> list[dict]:
    """Load up to `limit` recent chat traces for a user."""
    if redis is None:
        try:
            ddb_traces = _list_chat_traces_from_dynamodb(user_id, limit=limit)
        except Exception as exc:
            logger.warning("Failed to load DynamoDB chat traces for user=%s. %s", user_id, exc)
            ddb_traces = []
        if ddb_traces:
            logger.info(
                "Loaded %s conversation traces from DynamoDB for user=%s.",
                len(ddb_traces),
                user_id,
            )
            return ddb_traces

    client = redis or redis_client
    user_index_key = _user_conversation_index_key(user_id)

    try:
        conversation_ids = client.lrange(user_index_key, 0, max(0, limit - 1))
    except Exception as exc:
        logger.warning("Failed to list chat traces for user=%s. %s", user_id, exc)
        conversation_ids = []

    traces: list[dict] = []
    for conversation_id in conversation_ids:
        trace = _load_trace(conversation_id, redis=client)
        if not trace:
            continue
        if trace.get("user_id") != user_id:
            continue
        traces.append(trace)
    return traces


def clear_chat_traces(user_id: str, *, redis=None) -> dict[str, int]:
    """Delete all stored evaluation conversation traces for one user."""
    client = redis or redis_client
    normalized_user = str(user_id).strip()
    if not normalized_user:
        return {"trace_keys_deleted": 0, "index_key_deleted": 0}

    user_index_key = _user_conversation_index_key(normalized_user)
    try:
        conversation_ids = client.lrange(user_index_key, 0, -1)
    except Exception as exc:
        logger.warning("Failed reading chat trace index for user=%s. %s", normalized_user, exc)
        conversation_ids = []

    trace_keys: list[str] = []
    seen: set[str] = set()
    for conversation_id in conversation_ids:
        text = str(conversation_id).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        trace_keys.append(_conversation_key(text))

    trace_keys_deleted = 0
    index_key_deleted = 0
    try:
        if trace_keys:
            trace_keys_deleted = int(client.delete(*trace_keys))
        index_key_deleted = int(client.delete(user_index_key))
    except Exception as exc:
        logger.warning("Failed deleting chat traces for user=%s. %s", normalized_user, exc)

    return {
        "trace_keys_deleted": trace_keys_deleted,
        "index_key_deleted": index_key_deleted,
    }


def label_chat_trace(
    *,
    user_id: str,
    conversation_id: str,
    expected_answer: str | None = None,
    relevant_chunk_ids: list[str] | None = None,
    user_feedback: str | None = None,
    user_feedback_score: int | None = None,
    redis=None,
) -> dict | None:
    """Attach human labels used for stronger retrieval and generation evaluation."""
    client = redis or redis_client
    trace = _load_trace(conversation_id, redis=client)
    if not trace or trace.get("user_id") != user_id:
        return None

    labels = trace.get("labels")
    if not isinstance(labels, dict):
        labels = {}

    if expected_answer is not None:
        labels["expected_answer"] = expected_answer
    if relevant_chunk_ids is not None:
        labels["relevant_chunk_ids"] = [
            chunk_id.strip()
            for chunk_id in relevant_chunk_ids
            if isinstance(chunk_id, str) and chunk_id.strip()
        ]
    if user_feedback is not None:
        feedback = str(user_feedback).strip()
        labels["user_feedback"] = feedback or None
    if user_feedback_score is not None:
        try:
            score = int(user_feedback_score)
        except (TypeError, ValueError):
            score = None
        if score in {-1, 0, 1}:
            labels["user_feedback_score"] = score

    trace["labels"] = labels
    trace_key = _conversation_key(conversation_id)
    try:
        client.setex(trace_key, settings.memory.redis_ttl_seconds, _serialize_trace(trace))
    except Exception as exc:
        logger.warning("Failed to persist labels for conversation=%s. %s", conversation_id, exc)
        return None
    return trace


def evaluate_trace(trace: dict) -> dict:
    """Compute retrieval and generation metrics for one stored chat trace."""
    labels = trace.get("labels") if isinstance(trace.get("labels"), dict) else {}
    relevant_chunk_ids = labels.get("relevant_chunk_ids", []) if isinstance(labels, dict) else []
    expected_answer = labels.get("expected_answer") if isinstance(labels, dict) else None

    retrieved_results = trace.get("retrieved_results", [])
    if not isinstance(retrieved_results, list):
        retrieved_results = []

    retrieval_row: dict[str, float | int] = {}
    if relevant_chunk_ids:
        retrieval_row = retrieval_metrics(retrieved_results, relevant_chunk_ids)

    generation_row = generation_metrics(
        query=str(trace.get("prompt", "")),
        answer=str(trace.get("answer", "")),
        expected_answer=expected_answer if isinstance(expected_answer, str) else None,
        retrieved_results=retrieved_results,
    )
    evidence_urls = trace.get("evidence_urls", [])
    if not isinstance(evidence_urls, list):
        evidence_urls = []
    quality = trace.get("quality", {})
    quality = quality if isinstance(quality, dict) else {}
    citation_accuracy = quality.get("citation_accuracy")
    if citation_accuracy is None:
        citation_accuracy = citation_accuracy_score(
            answer=str(trace.get("answer", "")),
            evidence_urls=evidence_urls,
        )
    try:
        generation_row["citation_accuracy"] = float(citation_accuracy)
    except (TypeError, ValueError):
        generation_row["citation_accuracy"] = 0.0

    return {
        "retrieval": retrieval_row,
        "generation": generation_row,
    }


def _trace_source_count(trace: dict) -> int:
    retrieved_results = trace.get("retrieved_results", [])
    if not isinstance(retrieved_results, list):
        return 0
    sources: set[str] = set()
    for result in retrieved_results:
        if not isinstance(result, dict):
            continue
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        url = str(metadata.get("url", "")).strip()
        source_path = str(result.get("source_path", "")).strip()
        key = url or source_path
        if key:
            sources.add(key.lower())
    return len(sources)


def _web_fallback_summary(traces: list[dict], conversation_rows: list[dict]) -> dict:
    web_rows = [
        (trace, row)
        for trace, row in zip(traces, conversation_rows)
        if "web_fallback" in str(trace.get("retrieval_strategy", "")).lower()
    ]
    if not web_rows:
        return {
            "total_web_fallback_answers": 0,
            "avg_source_count": 0.0,
            "avg_groundedness": 0.0,
            "avg_citation_accuracy": 0.0,
            "feedback_count": 0,
            "avg_feedback_score": 0.0,
            "positive_feedback_rate": 0.0,
        }

    source_counts: list[float] = []
    groundedness_scores: list[float] = []
    citation_scores: list[float] = []
    feedback_scores: list[float] = []

    for trace, row in web_rows:
        source_counts.append(float(_trace_source_count(trace)))
        generation = row.get("metrics", {}).get("generation", {})
        if isinstance(generation, dict):
            groundedness_scores.append(float(generation.get("groundedness", 0.0)))
            citation_scores.append(float(generation.get("citation_accuracy", 0.0)))
        labels = trace.get("labels", {})
        labels = labels if isinstance(labels, dict) else {}
        feedback_score = labels.get("user_feedback_score")
        if feedback_score in {-1, 0, 1}:
            feedback_scores.append(float(feedback_score))

    positive = sum(1 for score in feedback_scores if score > 0)
    return {
        "total_web_fallback_answers": len(web_rows),
        "avg_source_count": mean(source_counts) if source_counts else 0.0,
        "avg_groundedness": mean(groundedness_scores) if groundedness_scores else 0.0,
        "avg_citation_accuracy": mean(citation_scores) if citation_scores else 0.0,
        "feedback_count": len(feedback_scores),
        "avg_feedback_score": mean(feedback_scores) if feedback_scores else 0.0,
        "positive_feedback_rate": (positive / len(feedback_scores)) if feedback_scores else 0.0,
    }


def get_user_evaluation_report(user_id: str, *, limit: int = 50, redis=None) -> dict:
    """Build an evaluation report over recent real conversations."""
    traces = list_chat_traces(user_id, limit=limit, redis=redis)
    conversation_rows = []
    retrieval_rows = []
    generation_rows = []
    labeled_count = 0

    for trace in traces:
        metrics = evaluate_trace(trace)
        if metrics["retrieval"]:
            retrieval_rows.append(metrics["retrieval"])
        if metrics["generation"]:
            generation_rows.append(metrics["generation"])

        labels = trace.get("labels", {})
        if isinstance(labels, dict) and (
            labels.get("expected_answer")
            or labels.get("relevant_chunk_ids")
            or labels.get("user_feedback")
            or labels.get("user_feedback_score") is not None
        ):
            labeled_count += 1

        conversation_rows.append(
            {
                "conversation_id": trace.get("conversation_id", ""),
                "created_at": trace.get("created_at", ""),
                "prompt": trace.get("prompt", ""),
                "answer": trace.get("answer", ""),
                "retrieval_strategy": trace.get("retrieval_strategy", ""),
                "retrieved_count": len(trace.get("retrieved_results", [])),
                "labels": labels if isinstance(labels, dict) else {},
                "metrics": metrics,
            }
        )

    return {
        "user_id": user_id,
        "total_conversations": len(traces),
        "labeled_conversations": labeled_count,
        "retrieval_metrics": aggregate_metric_rows(retrieval_rows),
        "generation_metrics": aggregate_metric_rows(generation_rows),
        "web_fallback_metrics": _web_fallback_summary(traces, conversation_rows),
        "conversations": conversation_rows,
    }
