"""SQS-backed async chat queue with DynamoDB job status tracking.

This module supports the async chat flow:
1) API enqueues a chat request to SQS,
2) worker consumes and runs model inference,
3) worker persists final job status/result in DynamoDB.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_JOB_STATUS_QUEUED = "queued"
_JOB_STATUS_PROCESSING = "processing"
_JOB_STATUS_COMPLETED = "completed"
_JOB_STATUS_FAILED = "failed"
_PUBLIC_JOB_ERROR = "Async chat job failed."
_TRACE_EVENT_MAX_PAYLOAD_CHARS = 800
_TRACE_MAX_EVENTS_PER_JOB = 120
_TRACE_APPEND_MAX_ATTEMPTS = 4
_TRACE_APPEND_BASE_DELAY_SECONDS = 0.05
DEBUG_ARTIFACT_DIR = os.getenv("UNIGRAPH_DEBUG_DIR", "data/debug/unigraph")
_VALID_STATUSES = {
    _JOB_STATUS_QUEUED,
    _JOB_STATUS_PROCESSING,
    _JOB_STATUS_COMPLETED,
    _JOB_STATUS_FAILED,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _utc_now().isoformat()


def _ttl_epoch_seconds(ttl_days: int) -> int:
    if ttl_days <= 0:
        return 0
    expires_at = _utc_now() + timedelta(days=ttl_days)
    return int(expires_at.timestamp())


def _region_name() -> str | None:
    region = (
        os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
    )
    return region or None


@lru_cache()
def _sqs_client():
    kwargs = {"region_name": _region_name()} if _region_name() else {}
    return boto3.client("sqs", **kwargs)


@lru_cache()
def _dynamodb_table():
    kwargs = {"region_name": _region_name()} if _region_name() else {}
    resource = boto3.resource("dynamodb", **kwargs)
    return resource.Table(settings.queue.llm_result_table.strip())


def _require_async_chat_config() -> None:
    if not settings.queue.llm_async_enabled:
        raise RuntimeError("Async chat queue is disabled.")
    if not settings.queue.llm_queue_url.strip():
        raise RuntimeError("LLM_QUEUE_URL is not configured.")
    if not settings.queue.llm_result_table.strip():
        raise RuntimeError("LLM_RESULT_TABLE is not configured.")


def _safe_message_group_id(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return "anonymous"
    return text[:128]


def _sanitize_job_error(error_message: str) -> str:
    """Reduce internal failure details to a client-safe, bounded message."""
    normalized = str(error_message).strip()
    lowered = normalized.lower()
    if not normalized:
        return _PUBLIC_JOB_ERROR
    if lowered.startswith("invalid async job payload"):
        return "Invalid async job payload."
    if lowered.startswith("queue enqueue failed"):
        return "Queue enqueue failed."
    return _PUBLIC_JOB_ERROR


def _safe_trace_value(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return Decimal(str(value))
    if isinstance(value, (str, int, bool)) or value is None:
        text = str(value) if value is not None else ""
        return text[:_TRACE_EVENT_MAX_PAYLOAD_CHARS] if isinstance(value, str) else value
    if isinstance(value, list):
        return [_safe_trace_value(item) for item in value[:20]]
    if isinstance(value, dict):
        sanitized: dict = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 30:
                break
            sanitized[str(key)[:80]] = _safe_trace_value(item)
        return sanitized
    return str(value)[:_TRACE_EVENT_MAX_PAYLOAD_CHARS]


def _sanitize_trace_event(event: dict) -> dict:
    raw = event if isinstance(event, dict) else {}
    event_type = str(raw.get("type", "")).strip() or "event"
    timestamp = str(raw.get("timestamp", "")).strip() or _now_iso()
    payload = raw.get("payload", {})
    safe_payload = _safe_trace_value(payload) if isinstance(payload, dict) else {}
    return {
        "type": event_type[:80],
        "timestamp": timestamp[:64],
        "payload": safe_payload,
    }


def _dynamodb_json_safe(value):
    """Convert debug payloads into DynamoDB-compatible JSON values."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str), parse_float=Decimal)
    except Exception:
        return {}


_DEBUG_MAX_JSON_CHARS = 120_000
_DEBUG_DEFAULT_LIST_LIMIT = 25
_DEBUG_LIST_LIMITS = {
    "raw_search_results": 3,
    "fan_out_search_results": 3,
    "chunks_created_detail": 0,
    "selected_evidence_chunks": 12,
    "evidence_passed_to_final_answer": 12,
    "evidence_passed_to_final_llm": 12,
    "field_mapped_evidence": 12,
    "excluded_evidence_chunks": 40,
    "excluded_chunks_with_reasons": 40,
    "skipped_urls": 60,
    "rejected_urls_with_reasons": 60,
    "rejected_pdfs": 40,
    "source_scores": 40,
}


def _debug_list_limit(key: str) -> int:
    return _DEBUG_LIST_LIMITS.get(str(key), _DEBUG_DEFAULT_LIST_LIMIT)


def _compact_debug_value(value, *, key: str = ""):
    if isinstance(value, dict):
        return {
            str(item_key): _compact_debug_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        limit = _debug_list_limit(key)
        if limit <= 0:
            return [{"_truncated": True, "omitted_count": len(value)}] if value else []
        compacted = [_compact_debug_value(item, key=key) for item in value[:limit]]
        if len(value) > limit:
            compacted.append({"_truncated": True, "omitted_count": len(value) - limit})
        return compacted
    if isinstance(value, str):
        max_chars = 700 if key in {"text", "snippet", "content", "chunks"} else 2000
        if len(value) > max_chars:
            return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"
        return value
    return value


def _compact_debug_info(debug_info: dict) -> dict:
    compacted = _compact_debug_value(debug_info)
    if isinstance(compacted, dict):
        compacted["_debug_truncated_for_storage"] = True
    try:
        serialized = json.dumps(compacted, ensure_ascii=False, default=str)
    except Exception:
        return {"_debug_truncated_for_storage": True, "error": "debug_not_serializable"}
    if len(serialized) <= _DEBUG_MAX_JSON_CHARS:
        return compacted
    summary_keys = [
        "request_id",
        "current_question",
        "original_question",
        "detected_intent",
        "detected_university",
        "detected_program",
        "resolved_official_domains",
        "retrieval_tier_used",
        "fallback_tier_used",
        "decomposition_fallback_used",
        "planner_type",
        "fallback_error",
        "generated_queries",
        "generated_search_queries",
        "selected_urls",
        "rejected_urls_with_reasons",
        "rejected_pdfs",
        "selected_evidence_chunks",
        "fields_missing_with_reason",
        "final_answer_shape",
        "final_confidence",
    ]
    summary = {
        key: _compact_debug_value(debug_info.get(key), key=key)
        for key in summary_keys
        if key in debug_info
    }
    summary["_debug_truncated_for_storage"] = True
    summary["_debug_original_size_chars"] = len(serialized)
    return summary


def _debug_artifact_dir() -> Path:
    return Path(os.getenv("UNIGRAPH_DEBUG_DIR", DEBUG_ARTIFACT_DIR)).expanduser()


def _write_debug_artifact(debug_info: dict) -> str:
    request_id = str(debug_info.get("request_id") or uuid4().hex).strip()
    safe_request_id = "".join(
        char if char.isalnum() or char in {"-", "_", ".", ":"} else "-" for char in request_id
    ).strip("-")
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    debug_dir = _debug_artifact_dir()
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{timestamp}-{safe_request_id or uuid4().hex}.json"
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(debug_info, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    logger.info("Async chat debug artifact written | path=%s", path)
    return str(path)


def _put_initial_job(job: dict) -> None:
    _dynamodb_table().put_item(
        Item=job,
        ConditionExpression="attribute_not_exists(job_id)",
    )


def _update_job(job_id: str, updates: dict) -> None:
    if not updates:
        return
    keys = []
    values = {}
    names = {}
    for idx, (key, value) in enumerate(updates.items(), start=1):
        name_key = f"#k{idx}"
        value_key = f":v{idx}"
        names[name_key] = key
        values[value_key] = value
        keys.append(f"{name_key}={value_key}")
    _dynamodb_table().update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET " + ", ".join(keys),
        ConditionExpression="attribute_exists(job_id)",
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def _normalized_mode(value: str | None) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"auto", "fast", "standard", "deep"}:
        return mode
    return "standard"


def enqueue_chat_job(
    *,
    user_id: str,
    prompt: str,
    session_id: str | None = None,
    mode: str | None = None,
    debug: bool = False,
) -> dict:
    """Create async chat job state and enqueue it into SQS."""
    _require_async_chat_config()

    normalized_user = str(user_id).strip()
    normalized_prompt = str(prompt).strip()
    normalized_session = str(session_id or normalized_user).strip() or normalized_user
    normalized_mode = _normalized_mode(mode)
    job_id = uuid4().hex
    now_iso = _now_iso()
    ttl_days = settings.queue.llm_result_ttl_days

    record = {
        "job_id": job_id,
        "request_id": job_id,
        "user_id": normalized_user,
        "session_id": normalized_session,
        "prompt": normalized_prompt,
        "mode": normalized_mode,
        "debug_enabled": bool(debug),
        "status": _JOB_STATUS_QUEUED,
        "answer": "",
        "error": "",
        "created_at": now_iso,
        "updated_at": now_iso,
        "started_at": "",
        "completed_at": "",
        "attempt_count": 0,
        "queue_message_id": "",
    }
    expires_at = _ttl_epoch_seconds(ttl_days)
    if expires_at > 0:
        record["expires_at"] = expires_at

    _put_initial_job(record)

    body = json.dumps(
        {
            "job_id": job_id,
            "user_id": normalized_user,
            "session_id": normalized_session,
            "prompt": normalized_prompt,
            "mode": normalized_mode,
            "debug": bool(debug),
            "submitted_at": now_iso,
        },
        ensure_ascii=False,
    )
    send_kwargs = {
        "QueueUrl": settings.queue.llm_queue_url.strip(),
        "MessageBody": body,
        # Fair-queue tenant key on Standard queue; also valid for FIFO.
        "MessageGroupId": _safe_message_group_id(normalized_session),
    }
    try:
        try:
            response = _sqs_client().send_message(**send_kwargs)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code not in {"InvalidParameterValue", "UnsupportedOperation"}:
                raise
            # Fallback for accounts/regions where MessageGroupId on Standard may not be enabled.
            send_kwargs.pop("MessageGroupId", None)
            response = _sqs_client().send_message(**send_kwargs)
    except Exception as exc:
        mark_job_failed(job_id, f"Queue enqueue failed: {exc}")
        raise

    message_id = str(response.get("MessageId", ""))
    _update_job(
        job_id,
        {
            "queue_message_id": message_id,
            "updated_at": _now_iso(),
        },
    )
    return {
        "job_id": job_id,
        "status": _JOB_STATUS_QUEUED,
        "submitted_at": now_iso,
    }


def get_chat_job(job_id: str) -> dict | None:
    """Fetch one async chat job result from DynamoDB."""
    if not job_id.strip() or not settings.queue.llm_result_table.strip():
        return None
    response = _dynamodb_table().get_item(Key={"job_id": job_id.strip()})
    item = response.get("Item")
    if not isinstance(item, dict):
        return None
    status = str(item.get("status", "")).strip().lower()
    if status not in _VALID_STATUSES:
        item["status"] = _JOB_STATUS_FAILED
    trace_events = item.get("trace_events", [])
    if not isinstance(trace_events, list):
        item["trace_events"] = []
    return item


def mark_job_processing(job_id: str) -> None:
    now_iso = _now_iso()
    _dynamodb_table().update_item(
        Key={"job_id": job_id},
        UpdateExpression=(
            "SET #status=:status, started_at=if_not_exists(started_at,:started_at), "
            "updated_at=:updated_at, trace_events=:trace_events ADD attempt_count :attempt_inc"
        ),
        ConditionExpression="attribute_exists(job_id)",
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues={
            ":status": _JOB_STATUS_PROCESSING,
            ":started_at": now_iso,
            ":updated_at": now_iso,
            ":trace_events": [],
            ":attempt_inc": 1,
        },
    )


def mark_job_completed(job_id: str, answer: str, debug_info: dict | None = None) -> None:
    now_iso = _now_iso()
    updates = {
        "status": _JOB_STATUS_COMPLETED,
        "answer": str(answer),
        "error": "",
        "completed_at": now_iso,
        "updated_at": now_iso,
    }
    if isinstance(debug_info, dict) and debug_info:
        updates["debug_artifact_path"] = _write_debug_artifact(debug_info)
    _update_job(job_id, updates)


def mark_job_failed(job_id: str, error_message: str) -> None:
    safe_error = _sanitize_job_error(error_message)
    _update_job(
        job_id,
        {
            "status": _JOB_STATUS_FAILED,
            "error": safe_error[:2000],
            "updated_at": _now_iso(),
        },
    )


def append_job_trace_event(job_id: str, event: dict) -> None:
    safe_event = _sanitize_trace_event(event)
    for attempt in range(1, _TRACE_APPEND_MAX_ATTEMPTS + 1):
        try:
            _dynamodb_table().update_item(
                Key={"job_id": job_id},
                UpdateExpression=(
                    "SET trace_events=list_append(if_not_exists(trace_events, :empty_events), :new_event), "
                    "updated_at=:updated_at"
                ),
                ConditionExpression=(
                    "attribute_exists(job_id) AND "
                    "(attribute_not_exists(trace_events) OR size(trace_events) < :max_events)"
                ),
                ExpressionAttributeValues={
                    ":empty_events": [],
                    ":new_event": [safe_event],
                    ":updated_at": _now_iso(),
                    ":max_events": _TRACE_MAX_EVENTS_PER_JOB,
                },
            )
            return
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", "")).strip()
            if error_code == "ConditionalCheckFailedException":
                # Trace append is best-effort: stop once the event cap is hit or record no longer matches.
                return
            if attempt >= _TRACE_APPEND_MAX_ATTEMPTS:
                raise
            if error_code not in {
                "ProvisionedThroughputExceededException",
                "ThrottlingException",
                "RequestLimitExceeded",
                "InternalServerError",
                "ServiceUnavailable",
                "LimitExceededException",
                "TransactionConflictException",
            }:
                raise
            delay = _TRACE_APPEND_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay + random.uniform(0, 0.03))
        except Exception:
            if attempt >= _TRACE_APPEND_MAX_ATTEMPTS:
                raise
            delay = _TRACE_APPEND_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay + random.uniform(0, 0.03))

    logger.warning(
        "AsyncLLMTraceAppendExhaustedRetries | job_id=%s | event_type=%s",
        job_id,
        str(safe_event.get("type", "event")),
    )


def receive_llm_job_messages() -> list[dict]:
    """Poll SQS for async chat jobs using long polling."""
    _require_async_chat_config()
    response = _sqs_client().receive_message(
        QueueUrl=settings.queue.llm_queue_url.strip(),
        MaxNumberOfMessages=settings.queue.llm_max_messages_per_poll,
        WaitTimeSeconds=settings.queue.llm_receive_wait_seconds,
        VisibilityTimeout=settings.queue.llm_visibility_timeout_seconds,
        MessageAttributeNames=["All"],
        AttributeNames=["All"],
    )
    messages = response.get("Messages", [])
    return messages if isinstance(messages, list) else []


def delete_llm_job_message(receipt_handle: str) -> None:
    """Acknowledge one successfully processed message from SQS."""
    if not receipt_handle:
        return
    _sqs_client().delete_message(
        QueueUrl=settings.queue.llm_queue_url.strip(),
        ReceiptHandle=receipt_handle,
    )
