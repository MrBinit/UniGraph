"""DynamoDB persistence for chat request metrics and rolling aggregate snapshots.

This module stores:
1) one request-level item per chat call in the configured requests table, and
2) a singleton aggregate snapshot item (`id=global`) in the configured aggregate table.

The request item includes selected top-level attributes for easy querying
(
    `session_id`,
    `outcome`,
    `latency_overall_ms`,
    `retrieval_strategy`,
    evidence counts,
    token usage,
)
plus a full-fidelity `record_json` payload.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from app.core.config import get_settings
from app.services.sqs_event_queue_service import (
    enqueue_evaluation_event,
    enqueue_metrics_aggregation_event,
)

settings = get_settings()
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _ttl_epoch_seconds(ttl_days: int) -> int:
    """Convert TTL days to a DynamoDB-compatible epoch-seconds expiry value."""
    expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)
    return int(expires_at.timestamp())


def _region_name() -> str | None:
    """Resolve AWS region from runtime environment variables."""
    region = (
        os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
    )
    return region or None


@lru_cache()
def _dynamodb_client():
    """Create and cache a low-level DynamoDB client."""
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "boto3 is required for DynamoDB metrics persistence."
        ) from exc
    kwargs = {"region_name": _region_name()} if _region_name() else {}
    return boto3.client("dynamodb", **kwargs)


def _int_str(value) -> str:
    """Safely coerce any numeric-like value to DynamoDB numeric-string format."""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def _float_str(value) -> str:
    """Safely coerce any numeric-like value to floating numeric-string format."""
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return "0.0"


def _persist_request_record(record: dict) -> None:
    """Write one request-level metrics item to the configured requests table."""
    table_name = settings.app.metrics_dynamodb_requests_table.strip()
    if not table_name:
        return

    request_id = str(record.get("request_id", "")).strip()
    if not request_id:
        return

    timings = record.get("timings_ms", {})
    if not isinstance(timings, dict):
        timings = {}
    retrieval = (
        record.get("retrieval", {}) if isinstance(record.get("retrieval"), dict) else {}
    )
    llm_usage = (
        record.get("llm_usage", {}) if isinstance(record.get("llm_usage"), dict) else {}
    )
    evidence = (
        retrieval.get("evidence", [])
        if isinstance(retrieval.get("evidence"), list)
        else []
    )
    item = {
        "request_id": {"S": request_id},
        "timestamp": {"S": str(record.get("timestamp", _now_iso()))},
        "user_id": {"S": str(record.get("user_id", ""))},
        "session_id": {"S": str(record.get("session_id") or record.get("user_id", ""))},
        "outcome": {"S": str(record.get("outcome", "unknown"))},
        "retrieval_strategy": {"S": str(retrieval.get("strategy", ""))},
        "retrieval_result_count": {"N": _int_str(retrieval.get("result_count"))},
        "retrieval_evidence_count": {"N": _int_str(len(evidence))},
        "retrieval_evidence_json": {
            "S": json.dumps(evidence, ensure_ascii=False, default=str)
        },
        "query": {"S": str(record.get("question", ""))},
        "answer": {"S": str(record.get("answer", ""))},
        "latency_overall_ms": {"N": _int_str(timings.get("overall_response_ms"))},
        "latency_llm_ms": {"N": _int_str(timings.get("llm_response_ms"))},
        "latency_short_term_ms": {"N": _int_str(timings.get("short_term_memory_ms"))},
        "latency_long_term_ms": {"N": _int_str(timings.get("long_term_memory_ms"))},
        "prompt_tokens": {"N": _int_str(llm_usage.get("prompt_tokens"))},
        "total_tokens": {"N": _int_str(llm_usage.get("total_tokens"))},
        "record_json": {"S": json.dumps(record, ensure_ascii=False, default=str)},
    }
    if settings.app.metrics_dynamodb_ttl_days > 0:
        item["expires_at"] = {
            "N": str(_ttl_epoch_seconds(settings.app.metrics_dynamodb_ttl_days))
        }

    _dynamodb_client().put_item(TableName=table_name, Item=item)


def _persist_aggregate_snapshot(aggregate: dict | None) -> None:
    """Write the latest aggregate metrics snapshot to the configured aggregate table."""
    table_name = settings.app.metrics_dynamodb_aggregate_table.strip()
    if not table_name or not isinstance(aggregate, dict):
        return

    latency_raw = aggregate.get("latency_ms", {})
    latency = latency_raw if isinstance(latency_raw, dict) else {}
    overall_latency = (
        latency.get("overall", {}) if isinstance(latency.get("overall"), dict) else {}
    )
    item = {
        "id": {"S": "global"},
        "updated_at": {"S": str(aggregate.get("updated_at", _now_iso()))},
        "total_requests": {"N": _int_str(aggregate.get("total_requests"))},
        "overall_avg_latency_ms": {"N": _float_str(overall_latency.get("average"))},
        "aggregate_json": {"S": json.dumps(aggregate, ensure_ascii=False, default=str)},
    }
    if settings.app.metrics_dynamodb_ttl_days > 0:
        item["expires_at"] = {
            "N": str(_ttl_epoch_seconds(settings.app.metrics_dynamodb_ttl_days))
        }

    _dynamodb_client().put_item(TableName=table_name, Item=item)


def persist_aggregate_snapshot_dynamodb(aggregate: dict | None) -> None:
    """Persist only the aggregate snapshot to DynamoDB.

    This is used by the background metrics aggregation queue worker.
    """
    if not settings.app.metrics_dynamodb_enabled:
        return
    _persist_aggregate_snapshot(aggregate)


def _queue_metrics_aggregation(record: dict, aggregate: dict | None) -> None:
    """Queue aggregate synchronization when enabled, with inline fallback."""
    request_id = str(record.get("request_id", "")).strip()
    queue_enabled = bool(settings.queue.metrics_aggregation_queue_enabled)
    queue_url = settings.queue.metrics_aggregation_queue_url.strip()
    if queue_enabled and queue_url and request_id:
        try:
            enqueue_metrics_aggregation_event(request_id)
            return
        except Exception as exc:
            logger.warning(
                (
                    "Metrics aggregation queue publish failed; "
                    "falling back to inline aggregate write. %s"
                ),
                exc,
            )
    _persist_aggregate_snapshot(aggregate)


def _queue_offline_evaluation(record: dict) -> None:
    """Queue per-request offline evaluation jobs for successful answers."""
    request_id = str(record.get("request_id", "")).strip()
    outcome = str(record.get("outcome", "")).strip().lower()
    if not request_id or outcome != "success":
        return
    if not settings.queue.evaluation_queue_enabled:
        return
    if not settings.queue.evaluation_queue_url.strip():
        return
    session_id = str(record.get("session_id") or record.get("user_id", "")).strip()
    try:
        enqueue_evaluation_event(request_id=request_id, session_id=session_id)
    except Exception as exc:
        logger.warning("Evaluation queue publish failed; continuing. %s", exc)


def persist_chat_metrics_dynamodb(record: dict, aggregate: dict | None = None) -> None:
    """Persist request-level and aggregate chat metrics into DynamoDB.

    This function is best-effort by design: write errors are logged and swallowed
    so observability failures do not block the chat response path.
    """
    if not settings.app.metrics_dynamodb_enabled:
        return

    try:
        _persist_request_record(record)
        _queue_metrics_aggregation(record, aggregate)
        _queue_offline_evaluation(record)
    except Exception as exc:
        logger.warning("DynamoDB metrics persistence failed; continuing. %s", exc)
